INPUT_TEMPLATE = """Plan actions for an Android phone task.
                                                                                                                                                    
<user_request>
{{ instruction }}
</user_request>
{% if device_date %}
<device_date>
{{ device_date }}
</device_date>
{%- endif %}
{%- if previous_plan %}
<previous_plan>
{{ previous_plan }}
</previous_plan>
{%- endif %}
{%- if previous_state %}
<previous_state>
{{ previous_state }}
</previous_state>
{%- endif %}
{%- if memory %}
<memory>
{{ memory }}
</memory>
{%- endif %}
{%- if last_thought %}
<last_thought>
{{ last_thought }}
</last_thought>
{%- endif %}
{%- if progress_summary %}
<progress_summary>
{{ progress_summary }}
</progress_summary>
{%- endif %}
{%- if action_history %}
<action_history>
{{ action_history }}
</action_history>
{%- endif %}
                                                                                                                                                    
<current_state>
{{ current_state }}
</current_state>

<atomic_actions>
Execute ONE action per turn using <action/>. Available actions:

- click(index): Click element at index.
  Example: <action type="click" index="5"/>

- long_press(index): Long press element at index.
  Example: <action type="long_press" index="3"/>

- type(text, index, clear): Type text into input field. Use index=-1 for focused field. Use clear="true" to clear existing text first.
  Example: <action type="type" text="Hello" index="5"/>
  Example: <action type="type" text="example.com" index="3" clear="true"/>

- system_button(button): Press system button (back, home, enter).
  Example: <action type="system_button" button="back"/>

- swipe(coordinate, coordinate2, duration): Swipe between points. Duration in seconds.
  Example: <action type="swipe" coordinate="[540, 1500]" coordinate2="[540, 500]" duration="1.0"/>

- open_app(text): Open app by name.
  Example: <action type="open_app" text="Settings"/>

- wait(duration): Wait for specified seconds.
  Example: <action type="wait" duration="2.0"/>
{%- if text_manipulation_enabled %}

- text_agent(task): Delegate text editing to specialized agent. Use for modifying text in focused input field.
  Example: <action type="text_agent" task="Add greeting at the beginning"/>
{%- endif %}
</atomic_actions>

<guidelines>
- Use `open_app` to open apps, not the app drawer
- Use search when looking for specific files/entries
- Store info in memory, not clipboard
- Match exact file names and dates from user request
- Don't do more than asked
</guidelines>
                                                                                                                                                    
---
                                                                                                                                                    
Analyze state and plan next steps.
                                                                                                                                                    
Output: <thought>, <add_memory>, <progress_summary>, <plan>, <action>
If complete: <answer success="true|false"> instead of <plan> and <action>"""
                                                                                                                                                    
OUTPUT_TEMPLATE = """
{%- if output_plan %}
<plan>
{{ output_plan }}
</plan>
{%- endif %}
{%- if output_action %}
{{ output_action }}
{%- endif %}
{%- if output_answer %}
<answer success="{{ 'true' if output_success else 'false' }}">
{{ output_answer }}
</answer>
{%- endif %}
{%- if output_add_memory %}
<add_memory>
{{ output_add_memory }}
</add_memory>
{%- endif %}
{%- if output_progress_summary %}
<progress_summary>
{{ output_progress_summary }}
</progress_summary>
{%- endif %}"""
                                                                                                                                                    
env = Environment(loader=BaseLoader())
input_template = env.from_string(INPUT_TEMPLATE)
output_template = env.from_string(OUTPUT_TEMPLATE)
                                     
print("Loading dataset...")
DATASET_PATH = "bountyhunterxx/gemini-3-distill-1k"
from datasets import load_dataset
                                                                                                               
# ============= LOAD DATASET =============
print("Loading dataset...")
raw_dataset = load_dataset(DATASET_PATH, split="train")
print(f"Total samples: {len(raw_dataset)}")
                                                                                                                                                    
def construct_prompt(row):
    return input_template.render(
        instruction=row.get('user_request', ''),
        device_date=row.get('device_date', ''),
        previous_plan=row.get('previous_plan', ''),
        previous_state=row.get('previous_state', ''),
        memory=row.get('memory', ''),
        last_thought=row.get('last_thought', ''),
        progress_summary=row.get('progress_summary', ''),
        action_history=row.get('action_history', ''),
        current_state=row.get('current_state', ''),
        text_manipulation_enabled=bool(row.get('text_manipulation', '')),
    ).strip()
                                                                                                                                                    
def construct_output(row):
    return output_template.render(
        output_plan=row.get('output_plan', ''),
        output_action=row.get('output_action', ''),
        output_answer=row.get('output_answer', ''),
        output_success=row.get('output_success', False),
        output_add_memory=row.get('output_add_memory', ''),
        output_progress_summary=row.get('output_progress_summary', ''),
    ).strip()
                                                                                                                                                    
def build_messages(row):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": construct_prompt(row)},
                {"type": "image", "image": row['image']}
            ]
        },
        {
            "role": "assistant",
            "reasoning_content": row.get('output_thought', ''),
            "content": [
                {"type": "text", "text": construct_output(row)}
            ]
        }
    ]
                                                                                                                                                    
# ============= TEST CHAT TEMPLATE =============
sample_idx = 5
row = raw_dataset[sample_idx]
messages = build_messages(row)
                                                                                                                                                    
print("\n=== Messages Structure ===")
print(f"User text: {messages[0]['content'][0]['text'][:200]}...")
print(f"\nAssistant reasoning_content: {messages[1]['reasoning_content'][:200]}...")
print(f"\nAssistant content: {messages[1]['content'][0]['text']}")
                                                                                                                                                    
# Apply chat template (this is where the bug likely manifests)
print("\n=== Applying Chat Template ===")
try:
    # For text-only test (remove image for tokenizer test)
    text_only_messages = [
        {"role": "user", "content": construct_prompt(row)},
        {"role": "assistant", "reasoning_content": row.get('output_thought', ''), "content": construct_output(row)}
    ]
    formatted = tokenizer.apply_chat_template(text_only_messages, tokenize=False, add_generation_prompt=False)
    print(formatted)
except Exception as e:
    print(f"Error applying chat template: {e}")