---
base_model: Qwen/Qwen3-8B
library_name: peft
license: apache-2.0
language:
- pt
tags:
- lora
- peft
- gguf
- game
- npc
- dialogue
- structured-output
pipeline_tag: text-generation
---

# Qwen3-8B NPC dialogue LoRA

LoRA adapter that makes Qwen3-8B play the NPCs of a Godot RPG: it answers in character (Brazilian Portuguese) and returns the JSON the game executes, `{"reply", "action", "mood_shift"}`.

On a held-out set of 392 turns (unseen worlds and characters), the action matches the reference **92.9%** of the time against 10.8% for the base model, with 100% valid JSON. Inside the game, the rule validator approved 97.5% of the replies.

Training code, metrics and the merge-vs-runtime-LoRA comparison: https://github.com/NicoDeVargas/qwen3-npc-dialogue-lora

## Files

- `adapter_model.safetensors`, `adapter_config.json`: PEFT adapter (r 16, alpha 32, all projections).
- `qwen3-8b-npc-dialogue-lora-f16.gguf`: the same adapter for llama.cpp.

## Use

```bash
llama-server -m Qwen3-8B-Q4_K_M.gguf --lora qwen3-8b-npc-dialogue-lora-f16.gguf -c 4096 --port 8080
```

Load the adapter at runtime. Merging it into the weights and requantizing to Q4 dropped action accuracy from 90% to 30% in our tests.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(model, "NicoDeVargas/qwen3-8b-npc-dialogue-lora")
```

Use the Qwen3 chat template with `enable_thinking=False`.
