<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="./static/droidrun-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/droidrun.png">
  <img src="./static/droidrun.png"  width="full">
</picture>

# ERNIE-4.5-VL Android Agent Fine-tune

This repository contains the fine-tuning setup for [baidu/ERNIE-4.5-VL-28B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-Thinking) to control Android devices through natural language commands.

Built on top of the [DroidRun](https://github.com/droidrun/droidrun) framework.

## Training Resources

| Resource | Description |
|----------|-------------|
| [training.jinja2](https://gist.github.com/Mariozada/71647a20aa70e38cad43564fe53c2b51) | Prompt template used for data collection |
| [training.ipynb](https://gist.github.com/Mariozada/1cd4f1f636a604f467ccd36305530e64) | Training notebook |
| [android_world](https://github.com/google-research/android_world) | Environment for data collection with built-in success validation |

## Model

- **Fine-tuned Model**: [ERNIE-4.5-VL-28B-A3B-PT-MOBILE](https://huggingface.co/fremko/ERNIE-4.5-VL-28B-A3B-PT-MOBILE)
- **Base Model**: [ERNIE-4.5-VL-28B-A3B-Thinking](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-Thinking)
- **Task**: Android device control via vision-language understanding
- **Framework**: DroidRun agent system

## Installation

```bash
pip install 'droidrun[google,anthropic,openai,deepseek,ollama,dev]'
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

## Usage

```bash
droidrun run "your command here" --vision
```

## Credits

- Original framework: [DroidRun](https://github.com/droidrun/droidrun)
- Data collection environment: [android_world](https://github.com/google-research/android_world) by Google Research

## License

MIT License - see the LICENSE file for details.
