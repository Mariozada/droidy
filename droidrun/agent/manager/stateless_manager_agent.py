"""
StatelessManagerAgent - Stateless planning agent that rebuilds context each turn.

This agent combines planning and execution - it creates a plan AND executes
the first action directly, without a separate Executor agent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel
from opentelemetry import trace

from droidrun.agent.common.events import RecordUIStateEvent, ScreenshotEvent
from droidrun.agent.manager.events import (
    ManagerActionResultEvent,
    ManagerContextEvent,
    ManagerPlanDetailsEvent,
    ManagerResponseEvent,
)
from droidrun.agent.manager.prompts import parse_manager_response
from droidrun.agent.usage import get_usage_from_response
from droidrun.agent.utils.chat_utils import to_chat_messages
from droidrun.agent.utils.inference import acall_with_retries
from droidrun.agent.utils.tracing_setup import record_langfuse_screenshot
from droidrun.agent.utils.prompt_resolver import PromptResolver
from droidrun.agent.utils.tools import (
    ATOMIC_ACTION_SIGNATURES,
    click,
    long_press,
    open_app,
    swipe,
    system_button,
    type,
    wait,
)
from droidrun.agent.oneflows.text_manipulator import run_text_manipulation_agent
from droidrun.config_manager.prompt_loader import PromptLoader

if TYPE_CHECKING:
    from droidrun.agent.droid import DroidAgentState
    from droidrun.config_manager.config_manager import AgentConfig, TracingConfig
    from droidrun.tools import Tools


logger = logging.getLogger("droidrun")


class StatelessManagerAgent(Workflow):
    def __init__(
        self,
        llm: LLM,
        tools_instance: "Tools | None",
        shared_state: "DroidAgentState",
        agent_config: "AgentConfig",
        custom_tools: dict = None,
        atomic_tools: dict = None,
        output_model: Type[BaseModel] | None = None,
        prompt_resolver: Optional[PromptResolver] = None,
        tracing_config: "TracingConfig | None" = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = agent_config.manager
        self.vision = self.config.vision
        self.tools_instance = tools_instance
        self.shared_state = shared_state
        self.custom_tools = custom_tools if custom_tools is not None else {}
        self.atomic_tools = (
            atomic_tools if atomic_tools is not None else ATOMIC_ACTION_SIGNATURES
        )
        self.output_model = output_model
        self.agent_config = agent_config
        self.prompt_resolver = prompt_resolver or PromptResolver()
        self.tracing_config = tracing_config

    def _build_action_history(self) -> list[dict]:
        if not self.shared_state.action_history:
            return []

        n = min(5, len(self.shared_state.action_history))
        return [
            {
                "action": act,
                "summary": summ,
                "outcome": outcome,
                "error": err,
            }
            for act, summ, outcome, err in zip(
                self.shared_state.action_history[-n:],
                self.shared_state.summary_history[-n:],
                self.shared_state.action_outcomes[-n:],
                self.shared_state.error_descriptions[-n:],
                strict=True,
            )
        ]

    async def _build_prompt(self, has_text_to_modify: bool) -> str:
        variables = {
            "instruction": self.shared_state.instruction,
            "device_date": await self.tools_instance.get_date(),
            "previous_plan": self.shared_state.previous_plan,
            "previous_state": self.shared_state.previous_formatted_device_state,
            "memory": self.shared_state.memory,
            "last_thought": self.shared_state.last_thought,
            "progress_summary": self.shared_state.progress_summary,
            "action_history": self._build_action_history(),
            "current_state": self.shared_state.formatted_device_state,
            "text_manipulation_enabled": has_text_to_modify,
            # Action execution variables
            "atomic_actions": {**self.atomic_tools, **self.custom_tools},
        }

        custom_prompt = self.prompt_resolver.get_prompt("manager_system")
        if custom_prompt:
            return PromptLoader.render_template(custom_prompt, variables)

        return await PromptLoader.load_prompt(
            self.agent_config.get_manager_system_prompt_path(),
            variables,
        )

    async def _validate_and_retry(
        self, messages: list[dict], initial_response: str
    ) -> str:
        output = initial_response
        parsed = parse_manager_response(output)

        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            error_message = None

            if parsed["answer"] and not parsed["plan"]:
                if parsed["success"] is None:
                    error_message = (
                        'You must include success="true" or success="false" attribute '
                        "in the <answer> or <request_accomplished> tag.\n"
                        'Example: <answer success="true">Task completed</answer>\n'
                        "Retry again."
                    )
                else:
                    break
            elif parsed["plan"] and parsed["answer"]:
                error_message = (
                    "You cannot include both <plan> and <answer> tags. "
                    "Use <answer> only when the task is complete.\n"
                    "Retry again."
                )
            elif not parsed["plan"] and not parsed["answer"]:
                error_message = (
                    "You must provide either a <plan> or an <answer>. "
                    "Please provide a plan with numbered steps."
                )
            else:
                break

            if error_message:
                retry_count += 1
                logger.warning(
                    f"Manager response invalid (retry {retry_count}/{max_retries}): {error_message}"
                )

                retry_messages = messages + [
                    {"role": "assistant", "content": [{"text": output}]},
                    {"role": "user", "content": [{"text": error_message}]},
                ]

                chat_messages = to_chat_messages(retry_messages)

                try:
                    response = await acall_with_retries(self.llm, chat_messages)
                    output = response.message.content
                    parsed = parse_manager_response(output)
                except Exception as e:
                    logger.error(f"LLM retry failed: {e}")
                    break

        return output

    @step
    async def prepare_context(
        self, ctx: Context, ev: StartEvent
    ) -> ManagerContextEvent:
        formatted_text, focused_text, a11y_tree, phone_state = (
            await self.tools_instance.get_state()
        )

        self.shared_state.previous_formatted_device_state = (
            self.shared_state.formatted_device_state
        )
        self.shared_state.formatted_device_state = formatted_text
        self.shared_state.focused_text = focused_text
        self.shared_state.a11y_tree = a11y_tree
        self.shared_state.phone_state = phone_state

        self.shared_state.update_current_app(
            package_name=phone_state.get("packageName", "Unknown"),
            activity_name=phone_state.get("currentApp", "Unknown"),
        )

        ctx.write_event_to_stream(RecordUIStateEvent(ui_state=a11y_tree))

        screenshot = None
        if self.vision or (
            hasattr(self.tools_instance, "save_trajectories")
            and self.tools_instance.save_trajectories != "none"
        ):
            try:
                result = await self.tools_instance.take_screenshot()
                if isinstance(result, tuple):
                    success, screenshot = result
                    if not success:
                        screenshot = None
                else:
                    screenshot = result

                if screenshot:
                    ctx.write_event_to_stream(ScreenshotEvent(screenshot=screenshot))
                    parent_span = trace.get_current_span()
                    record_langfuse_screenshot(
                        screenshot,
                        parent_span=parent_span,
                        screenshots_enabled=bool(
                            self.tracing_config
                            and self.tracing_config.langfuse_screenshots
                        ),
                        vision_enabled=self.vision,
                    )
            except Exception as e:
                logger.warning(f"Failed to capture screenshot: {e}")

        focused_text_clean = focused_text.replace("'", "").strip()
        has_text_to_modify = focused_text_clean != ""

        self.shared_state.has_text_to_modify = has_text_to_modify
        self.shared_state.screenshot = screenshot

        return ManagerContextEvent()

    @step
    async def get_response(
        self, ctx: Context, ev: ManagerContextEvent
    ) -> ManagerResponseEvent:
        has_text_to_modify = self.shared_state.has_text_to_modify
        screenshot = self.shared_state.screenshot

        prompt_text = await self._build_prompt(has_text_to_modify)
        messages = [{"role": "user", "content": [{"text": prompt_text}]}]

        if self.vision and screenshot:
            messages[0]["content"].append({"image": screenshot})

        chat_messages = to_chat_messages(messages)

        try:
            logger.info("[cyan]Manager response:[/cyan]")
            response = await acall_with_retries(
                self.llm, chat_messages, stream=self.agent_config.streaming
            )
            output = response.message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"Error calling LLM in stateless manager: {e}") from e

        usage = None
        try:
            usage = get_usage_from_response(self.llm.class_name(), response)
        except Exception as e:
            logger.warning(f"Could not get usage: {e}")

        output = await self._validate_and_retry(messages, output)

        event = ManagerResponseEvent(
            response=output,
            usage=usage,
            prompt_text=prompt_text,
            prompt_screenshot=screenshot if self.vision else None,
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def process_response(
        self, ctx: Context, ev: ManagerResponseEvent
    ) -> ManagerPlanDetailsEvent:
        output = ev.response
        parsed = parse_manager_response(output)

        self.shared_state.previous_plan = parsed["plan"]
        self.shared_state.last_thought = parsed["thought"]

        if parsed.get("progress_summary"):
            self.shared_state.progress_summary = parsed["progress_summary"]

        memory_update = parsed.get("memory", "").strip()
        if memory_update:
            if self.shared_state.memory:
                self.shared_state.memory += "\n" + memory_update
            else:
                self.shared_state.memory = memory_update

        self.shared_state.plan = parsed["plan"]
        self.shared_state.current_subgoal = parsed["current_subgoal"]
        self.shared_state.manager_answer = parsed["answer"]

        event = ManagerPlanDetailsEvent(
            plan=parsed["plan"],
            subgoal=parsed["current_subgoal"],
            thought=parsed["thought"],
            answer=parsed["answer"],
            memory_update=memory_update,
            progress_summary=parsed.get("progress_summary", ""),
            success=parsed["success"],
            full_response=output,
            action=parsed.get("action"),
        )
        ctx.write_event_to_stream(event)
        return event

    @step
    async def execute_action(
        self, ctx: Context, ev: ManagerPlanDetailsEvent
    ) -> ManagerActionResultEvent:
        """Execute the action if present."""
        if ev.action is None:
            # No action (answer case or missing action) - pass through
            return ManagerActionResultEvent(
                plan=ev.plan,
                subgoal=ev.subgoal,
                thought=ev.thought,
                answer=ev.answer,
                memory_update=ev.memory_update,
                progress_summary=ev.progress_summary,
                success=ev.success,
                full_response=ev.full_response,
                action=None,
                action_success=None,
                action_error="",
                action_summary="",
            )

        logger.debug(f"⚡ Executing action: {ev.action}")

        # Execute the action
        success, error, summary = await self._execute_action(ev.action)

        await asyncio.sleep(self.agent_config.after_sleep_action)

        logger.debug(f"{'✅' if success else '❌'} Execution complete: {summary}")

        return ManagerActionResultEvent(
            plan=ev.plan,
            subgoal=ev.subgoal,
            thought=ev.thought,
            answer=ev.answer,
            memory_update=ev.memory_update,
            progress_summary=ev.progress_summary,
            success=ev.success,
            full_response=ev.full_response,
            action=ev.action,
            action_success=success,
            action_error=error,
            action_summary=summary,
        )

    async def _execute_action(self, action_dict: dict) -> tuple[bool, str, str]:
        """Execute action and return (success, error, summary)."""
        action_type = action_dict.get("action", "unknown")

        # Check custom tools first
        if action_type in self.custom_tools:
            return await self._execute_custom_tool(action_type, action_dict)

        try:
            if action_type == "click":
                index = action_dict.get("index")
                if index is None:
                    return False, "Missing 'index' parameter", "Failed: click requires index"
                await click(index, tools=self.tools_instance)
                return True, "", f"Clicked element at index {index}"

            elif action_type == "long_press":
                index = action_dict.get("index")
                if index is None:
                    return False, "Missing 'index' parameter", "Failed: long_press requires index"
                success = await long_press(index, tools=self.tools_instance)
                if success:
                    return True, "", f"Long pressed element at index {index}"
                return False, "Long press failed", f"Failed to long press at index {index}"

            elif action_type == "type":
                text = action_dict.get("text")
                index = action_dict.get("index", -1)
                if text is None:
                    return False, "Missing 'text' parameter", "Failed: type requires text"
                await type(text, index, tools=self.tools_instance)
                return True, "", f"Typed '{text}' into element at index {index}"

            elif action_type == "system_button":
                button = action_dict.get("button")
                if button is None:
                    return False, "Missing 'button' parameter", "Failed: system_button requires button"
                result = await system_button(button, tools=self.tools_instance)
                if "Error" in result:
                    return False, result, f"Failed to press {button} button"
                return True, "", f"Pressed {button} button"

            elif action_type == "swipe":
                coordinate = action_dict.get("coordinate")
                coordinate2 = action_dict.get("coordinate2")
                duration = action_dict.get("duration", 1.0)

                if coordinate is None or coordinate2 is None:
                    return False, "Missing coordinate parameters", "Failed: swipe requires coordinates"

                if not isinstance(coordinate, list) or len(coordinate) != 2:
                    return False, f"Invalid coordinate: {coordinate}", "Failed: coordinate must be [x, y]"
                if not isinstance(coordinate2, list) or len(coordinate2) != 2:
                    return False, f"Invalid coordinate2: {coordinate2}", "Failed: coordinate2 must be [x, y]"

                success = await swipe(coordinate, coordinate2, duration, tools=self.tools_instance)
                if success:
                    return True, "", f"Swiped from {coordinate} to {coordinate2}"
                return False, "Swipe failed", f"Failed to swipe from {coordinate} to {coordinate2}"

            elif action_type == "wait":
                duration = action_dict.get("duration")
                if duration is None:
                    return False, "Missing 'duration' parameter", "Failed: wait requires duration"
                await wait(duration)
                return True, "", f"Waited for {duration} seconds"

            elif action_type == "open_app":
                text = action_dict.get("text")
                if text is None:
                    return False, "Missing 'text' parameter", "Failed: open_app requires text"
                await open_app(text, tools=self.tools_instance)
                return True, "", f"Opened app: {text}"

            elif action_type == "text_agent":
                task = action_dict.get("task")
                if task is None:
                    return False, "Missing 'task' parameter", "Failed: text_agent requires task"

                current_text = self.shared_state.focused_text or ""
                if not current_text.strip():
                    return False, "No focused text to edit", "Failed: text_agent requires focused text field"

                try:
                    text_to_type, code_ran = await run_text_manipulation_agent(
                        instruction=self.shared_state.instruction,
                        current_subgoal=task,
                        current_text=current_text,
                        overall_plan=self.shared_state.plan,
                        llm=self.tools_instance.text_manipulator_llm,
                        stream=self.agent_config.streaming,
                    )

                    if text_to_type and text_to_type.strip():
                        result = await self.tools_instance.input_text(
                            text_to_type, clear=True
                        )
                        if result and ("error" in result.lower() or "failed" in result.lower()):
                            return False, result, f"Text agent failed to input: {result}"
                        return True, "", f"Text agent modified text: {len(text_to_type)} chars"
                    else:
                        return False, "Text agent returned empty result", "Failed: no text modification"

                except Exception as e:
                    logger.error(f"Text agent error: {e}", exc_info=True)
                    return False, f"Text agent error: {str(e)}", f"Failed: text_agent error"

            else:
                return False, f"Unknown action type: {action_type}", f"Failed: unknown action '{action_type}'"

        except Exception as e:
            logger.error(f"Exception during action execution: {e}", exc_info=True)
            return False, f"Exception: {str(e)}", f"Failed to execute {action_type}: {str(e)}"

    async def _execute_custom_tool(
        self, action_type: str, action_dict: dict
    ) -> tuple[bool, str, str]:
        """Execute custom tool."""
        try:
            tool_spec = self.custom_tools[action_type]
            tool_func = tool_spec["function"]

            tool_args = {k: v for k, v in action_dict.items() if k != "action"}

            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(
                    **tool_args,
                    tools=self.tools_instance,
                    shared_state=self.shared_state,
                )
            else:
                result = tool_func(
                    **tool_args,
                    tools=self.tools_instance,
                    shared_state=self.shared_state,
                )

            summary = f"Executed custom tool '{action_type}'"
            if result is not None:
                summary += f": {str(result)}"

            return True, "", summary

        except TypeError as e:
            error_msg = f"Invalid arguments for custom tool '{action_type}': {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg, f"Failed: {action_type}"

        except Exception as e:
            error_msg = f"Error executing custom tool '{action_type}': {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return False, error_msg, f"Failed: {action_type}"

    @step
    async def finalize(
        self, ctx: Context, ev: ManagerActionResultEvent
    ) -> StopEvent:
        """Return manager results to parent workflow."""
        return StopEvent(
            result={
                "plan": ev.plan,
                "current_subgoal": ev.subgoal,
                "thought": ev.thought,
                "manager_answer": ev.answer,
                "memory_update": ev.memory_update,
                "success": ev.success,
                # Action execution results
                "action": ev.action,
                "action_success": ev.action_success,
                "action_error": ev.action_error,
                "action_summary": ev.action_summary,
            }
        )
