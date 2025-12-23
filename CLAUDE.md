# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

DroidRun is a powerful framework for controlling Android and iOS devices through LLM agents. It allows automation of device interactions using natural language commands, with support for multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama, DeepSeek).

## Architecture

### Agent System (Multi-Agent Workflow)

DroidRun uses a hierarchical agent architecture coordinated by `DroidAgent`:

1. **DroidAgent** (`droidrun/agent/droid/droid_agent.py`) - Main coordinator that orchestrates execution
   - **Reasoning Mode** (`reasoning=True`): Uses Manager � Executor workflow for complex planning
   - **Direct Mode** (`reasoning=False`): Uses CodeActAgent for immediate execution

2. **ManagerAgent** - High-level planning agent that creates and manages task plans
   - Analyzes current state and creates subgoals
   - Handles error escalation and plan adjustments
   - Can route to ScripterAgent via `<script>` tags

3. **ExecutorAgent** - Action execution agent that selects and executes atomic actions
   - Executes specific actions for each subgoal
   - Updates shared state with action outcomes

4. **CodeActAgent** (`droidrun/agent/codeact/codeact_agent.py`) - Direct execution agent
   - Generates and executes Python code using atomic actions
   - Used in non-reasoning mode or by DroidAgent for task execution
   - Supports safe execution mode (restricted imports/builtins)

5. **ScripterAgent** (`droidrun/agent/scripter/`) - Off-device Python script execution
   - Handles computations that don't require device interaction
   - Triggered when Manager includes `<script>...</script>` tags
   - Has separate max_steps and safe_execution config

6. **StructuredOutputAgent** (`droidrun/agent/oneflows/structured_output_agent.py`) - Structured data extraction
   - Extracts structured Pydantic models from text answers
   - Uses LLM.structured_predict() for parsing
   - Used when output_model is provided to DroidAgent

### Tools Architecture (2-Layer System)

Located in `droidrun/tools/`:

1. **Abstract Layer** (`tools.py`) - Defines the `Tools` abstract base class
2. **Implementation Layer**:
   - `adb.py` - `AdbTools` for Android devices via ADB
   - `ios.py` - `IOSTools` for iOS devices (limited functionality)
   - `portal_client.py` - Unified client for Android device communication (TCP or content provider mode)

**iOS Limitations:**
- `get_date()` method does not exist (not in IOSTools)
- `drag()` not implemented (returns False with message, does not raise exception)
- `input_text()` does not support `index` or `clear` parameters
- `long_press()` uses `_extract_element_coordinates_by_index()` which IS implemented

### Atomic Actions

The framework provides these core atomic actions (defined in `droidrun/agent/utils/tools.py`):
- `click(index)` - Tap UI element by index
- `long_press(index)` - Long press UI element
- `type(text, index)` - Input text into element
- `system_button(button)` - Press system buttons (back, home, enter)
- `swipe(coordinate, coordinate2)` - Swipe gesture
- `open_app(text)` - Open app by name

Additional tools:
- `remember(information)` - Store info in agent memory
- `complete(success, reason)` - Mark task as finished
- `get_state()` - Get accessibility tree + phone state
- `take_screenshot()` - Capture device screen

### Configuration System

Configuration is managed through YAML files using `DroidrunConfig.from_yaml()`:

**Loading Configuration:**
```python
from droidrun.config_manager import DroidrunConfig

# Load from file (uses PathResolver: working dir → package dir)
config = DroidrunConfig.from_yaml("config.yaml")

# Load from custom path
config = DroidrunConfig.from_yaml("custom/path/config.yaml")

# Load from environment variable
import os
config = DroidrunConfig.from_yaml(os.environ.get("DROIDRUN_CONFIG", "config.yaml"))
```

**Configuration Structure:**
- **Agent settings**: max_steps, reasoning mode, vision, prompts
- **LLM profiles**: Per-agent LLM configurations (manager, executor, codeact, text_manipulator, app_opener, scripter, structured_output)
- **Device settings**: serial, use_tcp, platform
- **Safe execution**: Import/builtin restrictions for CodeAct/Scripter
- **App Cards**: App-specific instruction cards (local/server/composite modes)
- **Credentials**: Secure credential storage via credential manager

**Path Resolution** (`PathResolver`):
- Automatically checks working directory first, then package directory
- Supports absolute and relative paths
- CLI resolution order: `--config` flag → `DROIDRUN_CONFIG` env var → `config.yaml`

**CLI Overrides:**
- CLI loads config, then applies overrides via direct mutation
- Modified config is passed to DroidAgent
- No singleton pattern - each load creates fresh config instance

### Credential Management

DroidRun includes a credential manager (`droidrun/credential_manager/`):
- YAML-based storage for secrets (file or in-memory modes)
- Integration with agent custom tools
- Credentials auto-injected as custom tools (e.g., `get_username()`, `get_password()`)
- Configured via `credentials` section in config.yaml
- Supports both enabled/disabled secrets and simple string format
- File path: `config/credentials_example.yaml` (template) or custom path via config

### Tracing and Telemetry

- **Arize Phoenix** integration for execution tracing (`droidrun/telemetry/phoenix.py`)
- **Langfuse** integration for LLM observability (`droidrun/agent/utils/tracing_setup.py`)
  - Supports custom user_id and session_id configuration
  - Session IDs are not persisted across runs (generated fresh each time)
  - Context propagation using `contextvars` for thread-safe tracing
- **Token Usage Tracking** - Manager, Executor, and Scripter agents track LLM token usage
- **Anonymous telemetry** with PostHog (opt-in via config)
- **Event tracking** - All actions emit events (TapActionEvent, SwipeActionEvent, etc.)
- **Trajectory saving** - Supports "none", "step", or "action" level recording

## Development Commands

### Setup and Installation

```bash
# Install with all providers
pip install 'droidrun[google,anthropic,openai,deepseek,ollama,dev]'

# Development setup
pip install -e ".[dev]"
```

### Running the CLI

```bash
# Basic usage (uses config.yaml or creates from config_example.yaml)
droidrun run "your command here"

# Common flags
droidrun run "command" --device SERIAL --provider GoogleGenAI --model models/gemini-2.5-flash
droidrun run "command" --vision --reasoning --debug --steps 20
droidrun run "command" --save-trajectory action --tracing
droidrun run "command" --ios  # Run on iOS device

# Device management
droidrun devices                    # List connected devices
droidrun connect 192.168.1.100:5555 # Connect over TCP/IP
droidrun disconnect SERIAL          # Disconnect device
droidrun setup --device SERIAL      # Install Portal APK
droidrun ping --device SERIAL       # Test Portal connection

# Macro management (record and replay action sequences)
droidrun macro record --name MACRO_NAME  # Record a macro
droidrun macro list                      # List saved macros
droidrun macro run MACRO_NAME            # Execute a macro
```

### Code Quality

```bash
# Format code
black droidrun/

# Lint code
ruff check droidrun/

# Type checking
mypy droidrun/

# Security checks
bandit -r droidrun/
safety scan
```

### Documentation

```bash
# Generate SDK reference docs (requires pydoc-markdown)
./gen-docs-sdk-ref.sh
```

## Key Files and Locations

- **Entry points**:
  - CLI: `droidrun/cli/main.py` (click-based CLI)
  - Main module: `droidrun/__main__.py`
  - Package script: `pyproject.toml` � `[project.scripts]` � `droidrun = "droidrun.cli.main:cli"`

- **Configuration**:
  - Default config: `droidrun/config.yaml` (or `config_example.yaml` as template)
  - Prompts: `droidrun/config/prompts/` (Jinja2 templates for each agent)
  - App cards: `droidrun/config/app_cards/app_cards.json`

- **Shared state**:
  - `droidrun/agent/droid/events.py` � `DroidAgentState` - Coordination state shared across agents
  - Tracks action history, visited packages/activities, error flags, scripter results
  - `droidrun/agent/utils/executer.py` � `ExecuterState` - State for code executor with UI state management

- **Utilities**:
  - `droidrun/agent/utils/async_utils.py` - Async tool wrapping with context propagation
  - `droidrun/agent/utils/executer.py` - Code execution engine with thread-safe context handling
  - `droidrun/agent/utils/tracing_setup.py` - Phoenix and Langfuse tracing setup
  - `droidrun/agent/usage.py` - Token usage tracking across agents

- **Portal App**:
  - Package: `com.droidrun.portal`
  - APK download: `droidrun/portal.py` � `download_portal_apk()`
  - Accessibility service must be enabled for full functionality

## Important Patterns

### Agent Workflow Events

DroidAgent uses llama-index workflows with custom events:
- `StartEvent` � triggers `start_handler()`
- `ManagerInputEvent` � triggers `run_manager()`
- `ManagerPlanEvent` � triggers `handle_manager_plan()`
- `ExecutorInputEvent` � triggers `run_executor()`
- `ExecutorResultEvent` � triggers `handle_executor_result()`
- `ScripterExecutorInputEvent` � triggers `run_scripter()`
- `CodeActExecuteEvent` � triggers `execute_task()`
- `FinalizeEvent` � triggers `finalize()` � `StopEvent`

Events are streamed to allow real-time monitoring and nested workflow execution.

### Tool Wrapping

Tools are wrapped for async contexts via `wrap_async_tools()` in `droidrun/agent/utils/async_utils.py` when running in direct execution mode. This allows synchronous tool execution in async workflows.

**Context Propagation**: Uses `contextvars` to propagate tracing context across thread boundaries when executing async tools in sync contexts. The global `_exec_context` variable stores the context for cross-thread execution.

### UI Action Decorator

Methods decorated with `@Tools.ui_action` automatically:
- Capture screenshots when `save_trajectories == "action"`
- Append to trajectory tracking lists
- Enable action replay and debugging

### Custom Tools

Custom tools can be added via:
1. **Credentials**: Auto-generated from credential manager
2. **User-defined**: Passed to `DroidAgent(custom_tools={...})`
3. **Built-in helpers**: `open_app` workflow using LLM

Format: `{"tool_name": {"signature": "...", "description": "...", "function": callable}}`

### Configuration Override Pattern

CLI flags override config values via direct mutation (see `droidrun/cli/main.py`):
```python
config = DroidrunConfig.from_yaml(config_path or "config.yaml")
if vision is not None:
    config.agent.manager.vision = vision
    config.agent.executor.vision = vision
```

### Context Manager for Cloud Devices

Always use context manager for cloud devices to ensure termination:
```python
with CloudAdbTools(api_client=client, apps=["com.example.app"]) as tools:
    tools.start_app("com.example.app")
    tools.tap_by_index(5)
# Device automatically terminated
```

## Testing Approach

This project does not currently have a formal test suite. Testing is done via:
1. Manual testing with real devices
2. CLI test commands (see `droidrun/cli/main.py` � `test()` function)
3. Security checks (bandit, safety)

## Recent Updates (Last Updated: 2025-11-01)

### Tracing & Context Management (Recent)
- **Langfuse Integration**: Added Langfuse tracing support with configurable user_id and session_id
- **Context Propagation**: Implemented `contextvars`-based context propagation for thread-safe tracing across async/sync boundaries (`droidrun/agent/utils/async_utils.py`, `droidrun/agent/utils/executer.py`)
- **Session Management**: Session IDs are now generated fresh per run and not persisted

### Executor Improvements (Recent)
- **Event Loop Handling**: Removed `loop` parameter from `SimpleCodeExecutor` constructor - now uses `asyncio.get_running_loop()` at execution time
- **Thread Safety**: Enhanced thread-safe execution with proper context propagation for tracing
- **Pydantic v2**: Updated `ExecuterState` to use `model_config = ConfigDict(arbitrary_types_allowed=True)`

### Token Usage Tracking (Recent)
- Added token usage tracking in `ManagerAgent`, `ExecutorAgent`, and `ScripterAgent`
- Usage events now include token consumption metrics
- Updated events: `ManagerPlanEvent`, `ExecutorResultEvent`, `ScripterResultEvent` now contain usage data

### Configuration Changes (Recent)
- New Langfuse configuration options in `config_example.yaml`
- Added `OpenAILike` as a provider option for usage tracking
- Added `structured_output` LLM profile for structured data extraction

### Configuration System Refactor (2025-11-01)
- **Removed ConfigManager singleton**: Replaced with simple `DroidrunConfig.from_yaml()` function
- **Integrated PathResolver**: `from_yaml()` now uses PathResolver by default
- **Simplified config loading**: No more singleton pattern, each load is explicit and fresh
- **Cleaner CLI**: Direct config loading without manager wrapper
- **Net result**: -262 lines of code, simpler architecture

### Documentation Corrections (2025-11-01)
- Added missing `StructuredOutputAgent` documentation
- Corrected iOS `get_date()` limitation (method doesn't exist, not just unimplemented)
- Corrected iOS `drag()` behavior (returns False, doesn't raise exception)
- Added missing `--ios` CLI flag documentation
- Added missing `macro` CLI command group documentation

## Common Gotchas

1. **Portal APK required**: The DroidRun Portal app must be installed and accessibility enabled on devices
2. **TCP vs Content Provider**: TCP is faster but requires port forwarding; content provider is fallback
3. **Vision mode costs**: Enabling vision sends screenshots to LLM (increases token usage)
4. **Max steps**: Default is 15; increase for complex tasks via `--steps` or `agent.max_steps` in config
5. **Safe execution**: When enabled, restricts imports/builtins (see `safe_execution` in config)
6. **Credentials**: Stored in YAML format at `config/credentials_example.yaml` (template)
7. **Config path resolution**: `from_yaml()` uses PathResolver - checks working dir first, then package dir
8. **Config parse failures**: Silent fallback to defaults with warning if YAML parsing fails - check logs for config errors
9. **Executor initialization**: Do NOT pass `loop` parameter to `SimpleCodeExecutor` - it's deprecated and automatically uses the running event loop
10. **Context propagation**: When wrapping async tools for sync execution, context is automatically propagated using `contextvars` for tracing

## Security Notes

- Security checks use `bandit` and `safety` (see README.md)
- Safe execution mode restricts dangerous operations (os, subprocess, eval, exec, etc.)
- Credentials are stored in YAML (not encrypted) - ensure `credentials.yaml` is in `.gitignore`
- Never commit API keys or sensitive data (use .env files or credential manager)
- PathResolver expands `~` to user home directory - be cautious with user-provided paths
- Safe execution blocks dangerous modules by default (see `safe_execution.blocked_modules` in config)

## Community and Support

- Documentation: https://docs.droidrun.ai
- Discord: https://discord.gg/ZZbKEZZkwK
- GitHub Issues: https://github.com/droidrun/droidrun/issues
- Twitter/X: https://x.com/droid_run
- Benchmark: https://droidrun.ai/benchmark




# LlamaIndex `structured_predict()` Guide

## Overview

`structured_predict()` is a method on every LLM that takes a Pydantic class, a prompt template, and variables - then returns a structured Pydantic object. It handles function calling or text parsing automatically.

---

## Basic Usage

```python
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

# 1. Define your output schema
class Album(BaseModel):
    """An album with songs."""
    name: str
    artist: str
    songs: List[str]

# 2. Create your LLM
llm = OpenAI(model="gpt-4o")

# 3. Define your prompt with variables
prompt = PromptTemplate(
    "Generate an album inspired by {movie_name} with {num_songs} songs"
)

# 4. Call structured_predict
album = llm.structured_predict(
    Album,                    # Output class
    prompt,                   # Prompt template
    movie_name="Inception",   # Variable values
    num_songs=5
)

# album is now an Album object
print(album.name)
print(album.artist)
print(album.songs)
```

---

## Signature

```python
llm.structured_predict(
    output_cls: Type[Model],           # Your Pydantic class
    prompt: PromptTemplate,            # Prompt with {variables}
    llm_kwargs: Optional[Dict] = None, # Additional LLM parameters
    **prompt_args                      # Values for template variables
) -> Model
```

**Returns**: Instance of your Pydantic class

---

## Variants

### Async Version
```python
album = await llm.astructured_predict(
    Album,
    prompt,
    movie_name="Interstellar",
    num_songs=6
)
```

### Streaming Version
Returns partial objects as they're generated:

```python
stream = llm.stream_structured_predict(
    Album,
    prompt,
    movie_name="The Matrix",
    num_songs=8
)

for partial_album in stream:
    print(partial_album.name)  # Updates as streaming progresses
```

### Async Streaming Version
```python
stream = llm.astream_structured_predict(
    Album,
    prompt,
    movie_name="Blade Runner",
    num_songs=7
)

async for partial_album in stream:
    print(partial_album.name)
```

---

## Prompt Templates

### Simple Template
```python
prompt = PromptTemplate("Extract an invoice from: {text}")

invoice = llm.structured_predict(Invoice, prompt, text=document_text)
```

### Multi-Variable Template
```python
prompt = PromptTemplate(
    "Extract {data_type} from the following {source_type}: {content}"
)

result = llm.structured_predict(
    Contact,
    prompt,
    data_type="contact information",
    source_type="email",
    content=email_body
)
```

### Complex Instructions
```python
prompt = PromptTemplate(
    "Analyze the following text and extract key information.\n"
    "Text: {text}\n"
    "Focus on: {focus_areas}\n"
    "If you cannot find {field}, use {default_value} instead."
)

data = llm.structured_predict(
    Report,
    prompt,
    text=document,
    focus_areas="financial metrics",
    field="revenue",
    default_value="N/A"
)
```

---

## Common Patterns

### Extract from Unstructured Text
```python
class Invoice(BaseModel):
    """Invoice information."""
    invoice_id: str
    date: str
    total: float
    line_items: List[str]

prompt = PromptTemplate("Extract invoice details from: {text}")

invoice = llm.structured_predict(Invoice, prompt, text=pdf_text)
```

### Transform Data Format
```python
class StructuredEmail(BaseModel):
    """Structured email data."""
    sender: str
    subject: str
    key_points: List[str]
    action_items: List[str]

prompt = PromptTemplate(
    "Convert this email to structured format: {email_text}"
)

structured = llm.structured_predict(
    StructuredEmail,
    prompt,
    email_text=raw_email
)
```

### Conditional Extraction
```python
prompt = PromptTemplate(
    "Extract {entity_type} from: {text}\n"
    "If you cannot find a {field_name}, use '{default}' as the value."
)

entity = llm.structured_predict(
    Entity,
    prompt,
    entity_type="company information",
    text=article,
    field_name="founding date",
    default="Unknown"
)
```

### Multiple Context Sources
```python
class Summary(BaseModel):
    """Document summary."""
    main_points: List[str]
    sentiment: str
    category: str

prompt = PromptTemplate(
    "Summarize this document:\n"
    "Title: {title}\n"
    "Content: {content}\n"
    "Author: {author}\n"
    "Date: {date}"
)

summary = llm.structured_predict(
    Summary,
    prompt,
    title=doc.title,
    content=doc.text,
    author=doc.author,
    date=doc.date
)
```

---

## Best Practices

### 1. Always Add Descriptions
```python
class Product(BaseModel):
    """Product information extracted from text."""
    name: str = Field(description="The product name")
    price: float = Field(description="Price in USD")
    category: str = Field(description="Product category")
```

The LLM uses these descriptions to understand what to extract.

### 2. Use Clear Variable Names
```python
# Good
prompt = PromptTemplate("Extract data from {document_text}")

# Avoid
prompt = PromptTemplate("Extract data from {x}")
```

### 3. Provide Context in Prompt
```python
prompt = PromptTemplate(
    "You are extracting structured data from a {doc_type}.\n"
    "Extract the following information: {text}\n"
    "Be precise and follow the schema exactly."
)
```

### 4. Handle Missing Data
```python
prompt = PromptTemplate(
    "Extract information from: {text}\n"
    "If any field is not found, use 'Not specified' as the value."
)
```

---

## When to Use `structured_predict()`

✅ **Use when:**
- You need custom prompts for each call
- Different output types for different calls
- One-off structured extractions
- You want control over the prompt template

❌ **Don't use when:**
- You're making many calls with the same output type (use `as_structured_llm()` instead)
- You need to integrate with query engines (pass structured LLM to query engine)

---

## Tips

- The LLM automatically handles function calling vs text parsing
- Works with any LLM (OpenAI, Anthropic, local models, etc.)
- Prompt variables must match the `{placeholders}` in your template
- Return type is always an instance of your Pydantic class
- Use streaming variants for long-running extractions to see progress