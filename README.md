<div align="center">

[![Demo Video](https://img.youtube.com/vi/94Qmho1VLCs/hqdefault.jpg)](https://www.youtube.com/watch?v=94Qmho1VLCs)

**Click to watch demo**

</div>

# ERNIE-4.5-VL Android Agent Fine-tune

This repository contains the fine-tuning setup for [baidu/ERNIE-4.5-VL-28B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-Thinking) to control Android devices through natural language commands.

Built on top of the [DroidRun](https://github.com/droidrun/droidrun) framework.

## Training Resources

| Resource | Description |
|----------|-------------|
| [training.jinja2](https://gist.github.com/Mariozada/71647a20aa70e38cad43564fe53c2b51) | Prompt template used for data collection |
| [training.ipynb](https://gist.github.com/Mariozada/1cd4f1f636a604f467ccd36305530e64) | Training notebook |
| [manager-merge-distill-1k](https://huggingface.co/datasets/fremko/manager-merge-distill-1k) | Original training dataset |
| [manager-merge-distill-1k-cut-10](https://huggingface.co/datasets/fremko/manager-merge-distill-1k-cut-10) | Training dataset (top 10% long context removed to fit) |
| [android_world](https://github.com/google-research/android_world) | Environment for data collection with built-in success validation |

## Model

- **Fine-tuned Model**: [ERNIE-4.5-VL-28B-A3B-PT-MOBILE](https://huggingface.co/fremko/ERNIE-4.5-VL-28B-A3B-PT-MOBILE)
- **Base Model**: [ERNIE-4.5-VL-28B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-Thinking)
- **Task**: Android device control via vision-language understanding
- **Framework**: DroidRun agent system



## Hosting

```bash
uv pip install -U vllm
uv pip install decoder
vllm serve "fremko/ERNIE-4.5-VL-28B-A3B-PT-MOBILE" \
    --max-model-len 32768 \
    --port 8000 \
    --generation-config vllm \
    --override-generation-config '{"temperature": 0.0, "top_p": 1.0}' \
    --dtype bfloat16 \
    --mm-processor-kwargs '{"min_pixels": 200704, "max_pixels": 5017600}' \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code
```

## Installation

**Requirements:** An Android emulator running or a physical Android device connected via ADB.

```bash
# Install the droidrun package with Google provider support
uv pip install git+https://github.com/Mariozada/droidy[google]

# Install Portal APK on the connected device/emulator (requires running emulator or connected device)
droidrun setup

# Install OpenAI-like LLM provider for ERNIE model compatibility
uv pip install llama-index-llms-openai-like

# Download the pre-configured config file
wget https://raw.githubusercontent.com/Mariozada/droidy/main/config.yaml
```

## Configuration

In `config.yaml`, replace the manager's `api_base` URL with your hosted model endpoint:

```yaml
llm_profiles:
  manager:
    provider: OpenAILike
    model: fremko/ERNIE-4.5-VL-28B-A3B-PT-MOBILE
    api_base: https://your-hosted-endpoint.com/v1  # Replace with your URL
```

> **Note:** The `config.yaml` contains multiple agent profiles configured with Google GenAI and one using ERNIE. However, only two agents are actually used during execution: **manager** and **app_opener**. The manager uses our fine-tuned ERNIE model, while the app_opener currently uses Google GenAI. The app_opener can also be switched to use ERNIE - we simply didn't update it due to our late submission deadline.

## Usage

```bash
droidrun run "your command here" --config config.yaml
```

## Credits

- Original framework: [DroidRun](https://github.com/droidrun/droidrun)
- Data collection environment: [android_world](https://github.com/google-research/android_world) by Google Research

## License

MIT License - see the LICENSE file for details.
