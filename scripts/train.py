import json, math, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = "Qwen/Qwen3-8B"
WORK = os.environ.get("WORK_DIR", "/opt/dlami/nvme/work")
OUT = os.path.join(WORK, "lora")
MAX_LEN = 2048

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or "<|endoftext|>"


def encode(row):
    msgs = row["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    target = msgs[-1]["content"] + "<|im_end|>\n"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"]
    ids = (p + t)[:MAX_LEN]
    labels = ([-100] * len(p) + t)[:MAX_LEN]
    return {"input_ids": ids, "labels": labels}


def load(path):
    return [encode(json.loads(l)) for l in open(path, encoding="utf-8")]


train = load(os.path.join(WORK, "train.jsonl"))
test = load(os.path.join(WORK, "test.jsonl"))
lens = sorted(len(r["input_ids"]) for r in train)
print("exemplos", len(train), "teste", len(test), "tokens p50", lens[len(lens) // 2], "p99", lens[int(len(lens) * 0.99)], "max", lens[-1], "cortados", sum(1 for x in lens if x >= MAX_LEN), flush=True)


class DS(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}


bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
QUANT = os.environ.get("QUANT", "0") == "1"
if QUANT:
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
else:
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=14,
    weight_decay=0.0,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    train_sampling_strategy="group_by_length",
    report_to=[],
    seed=42,
    remove_unused_columns=False,
    use_liger_kernel=True,
)
trainer = Trainer(model=model, args=args, train_dataset=DS(train), eval_dataset=DS(test), data_collator=collate)
t0 = time.time()
trainer.train()
trainer.save_model(OUT + "/final")
tok.save_pretrained(OUT + "/final")
json.dump(trainer.state.log_history, open(OUT + "/log_history.json", "w"), indent=1)
print("fim em", round((time.time() - t0) / 60, 1), "min", flush=True)
