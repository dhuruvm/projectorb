---
name: Orb Dependencies
description: How deps are installed for the Orb/GPT-2 project — quirks and what's been set up
---

## Install method
`uv pip install` NOT `uv sync` — the pyproject.toml has complex pytorch-cpu index sources that cause uv sync to fail for transformers on Linux.

```bash
uv pip install transformers torch peft datasets gguf sentencepiece
```

## pyproject.toml quirk
`requires-python = ">=3.13,<3.14"` — had to cap at 3.14 because uv couldn't resolve transformers for the Python 3.14 split marker on Linux.

## What's installed
- gradio>=6.20 (in pyproject.toml)
- transformers, torch (CPU), peft, datasets — installed via uv pip install
- gguf, sentencepiece — needed for GGUF conversion (installed by convert_to_gguf.py)

## Model weights
GPT-2 weights downloaded programmatically via:
```python
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.save_pretrained('models/gpt2')
```
The models/gpt2/ directory had tokenizer files but was missing model.safetensors — this caused a startup crash that was fixed by downloading weights.

## Workflow name
"GPT-2 Gradio App" — kept this name even though the model is now called Orb (it's just a workflow label, not user-facing).
