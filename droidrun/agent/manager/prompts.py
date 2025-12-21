"""
Prompts for the ManagerAgent.
"""

import re
import xml.etree.ElementTree as ET
import json as json_module


def parse_action_tag(response: str) -> dict | None:
    """
    Extract and parse <action .../> tag from response.

    Supports both self-closing <action type="click" index="5"/>
    and full <action type="click" index="5"></action> formats.

    Returns:
        Dict with action parameters, e.g. {"action": "click", "index": 5}
        or None if no valid action tag found.
    """
    start = response.find("<action")
    if start == -1:
        return None

    # Find the end - either /> or </action>
    self_close = response.find("/>", start)
    full_close = response.find("</action>", start)

    if self_close != -1 and (full_close == -1 or self_close < full_close):
        end = self_close + 2
    elif full_close != -1:
        end = full_close + len("</action>")
    else:
        return None

    action_str = response[start:end]

    try:
        element = ET.fromstring(action_str)
        attrs = element.attrib

        # Convert "type" to "action" for consistency with executor
        action_dict = {"action": attrs.pop("type", "unknown")}

        # Convert attribute values to appropriate types
        for k, v in attrs.items():
            # Handle coordinate arrays like "[540, 1500]"
            if v.startswith("[") and v.endswith("]"):
                try:
                    action_dict[k] = json_module.loads(v)
                except (json_module.JSONDecodeError, ValueError):
                    action_dict[k] = v
            # Handle integers (including negative)
            elif v.lstrip("-").isdigit():
                action_dict[k] = int(v)
            # Handle floats
            elif v.replace(".", "", 1).replace("-", "", 1).isdigit():
                action_dict[k] = float(v)
            else:
                action_dict[k] = v

        # Store original XML for display in action history
        action_dict["_xml"] = action_str

        return action_dict
    except ET.ParseError:
        return None


def parse_manager_response(response: str) -> dict:
    """
    Parse manager LLM response into structured dict.

    Extracts XML-style tags from the response:
    - <thought>...</thought>
    - <add_memory>...</add_memory>
    - <plan>...</plan>
    - <request_accomplished success="true|false">...</request_accomplished> (answer)

    Also derives:
    - current_subgoal: first line of plan (with list markers removed)
    - If first item is <script> tag, extract script content as current_subgoal
    - success: bool | None parsed from request_accomplished success attribute

    Args:
        response: Raw LLM response text

    Returns:
        Dict with keys:
            - thought: str
            - memory: str
            - plan: str
            - current_subgoal: str (first line of plan, cleaned, or script content)
            - answer: str (from request_accomplished tag)
            - success: bool | None (True/False if task complete, None if still in progress)
    """

    def extract(tag: str) -> str:
        """Extract content between XML-style tags (handles attributes)."""
        # Use regex to handle tags with attributes like <tag attr="value">
        pattern = rf"<{tag}(?:\s+[^>]*)?>(.+?)</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    thought = extract("thought")
    memory_section = extract("add_memory")
    plan = extract("plan")
    progress_summary = extract("progress_summary")
    answer = extract("request_accomplished")

    # Also support <answer> tag as alternative to <request_accomplished>
    if not answer:
        answer = extract("answer")

    # Parse success attribute from request_accomplished or answer tag
    success = None
    if answer:
        # Try to extract success attribute from tag
        success_match = re.search(
            r'<(?:request_accomplished|answer)\s+success="(true|false)">', response
        )
        if success_match:
            success = success_match.group(1) == "true"
        else:
            # Default to True for backward compatibility if no attribute present
            success = True

    # Parse current subgoal from first line of plan
    current_goal_text = plan

    # Check if first item is a <script> tag
    script_match = re.search(
        r"^\s*<script>(.*?)</script>", current_goal_text, re.DOTALL
    )

    if script_match:
        # Script is first task - extract script content with tag
        current_subgoal = f"<script>{script_match.group(1).strip()}</script>"
    else:
        # Regular subgoal - use existing logic
        plan_lines = [
            line.strip() for line in current_goal_text.splitlines() if line.strip()
        ]
        if plan_lines:
            first_line = plan_lines[0]
        else:
            first_line = current_goal_text.strip()

        # Remove common list markers like "1.", "-", "*", or bullet characters
        first_line = re.sub(
            r"^\s*\d+\.\s*", "", first_line
        )  # Remove "1. ", "2. ", etc.
        first_line = re.sub(r"^\s*[-*]\s*", "", first_line)  # Remove "- " or "* "
        first_line = re.sub(r"^\s*•\s*", "", first_line)  # Remove bullet "• "

        current_subgoal = first_line.strip()

    # Parse action tag (for stateless manager with direct execution)
    action = parse_action_tag(response)

    return {
        "thought": thought,
        "plan": plan,
        "memory": memory_section,
        "current_subgoal": current_subgoal,
        "answer": answer,
        "success": success,
        "progress_summary": progress_summary,
        "action": action,
    }
