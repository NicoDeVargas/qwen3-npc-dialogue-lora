import json, os, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "Qwen/Qwen3-8B"
ADAPTER = os.environ.get("ADAPTER", "")
LIMIT = int(os.environ.get("LIMIT", "0"))
OUT = sys.argv[1]

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "left"
tok.pad_token = tok.pad_token or "<|endoftext|>"
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
model.eval()

rows = [json.loads(l) for l in open(os.environ.get("TEST_FILE", "/opt/dlami/nvme/work/test.jsonl"), encoding="utf-8")]
if LIMIT:
    rows = rows[:LIMIT]


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        d = json.loads(m.group(0) if m else text)
        if isinstance(d, dict) and isinstance(d.get("reply"), str) and isinstance(d.get("action"), dict) and d.get("mood_shift") in (-1, 0, 1):
            return d
    except Exception:
        pass
    return None


results = []
t0 = time.time()
B = 12
for i in range(0, len(rows), B):
    chunk = rows[i:i + B]
    prompts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in chunk]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
    for r, o in zip(chunk, out):
        text = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        gold = json.loads(r["messages"][-1]["content"])
        got = parse(text)
        results.append({"meta": r.get("meta", {}), "gold": gold, "raw": text, "ok_json": got is not None, "got": got})
    print(i + len(chunk), "de", len(rows), round(time.time() - t0), "s", flush=True)

ok = [x for x in results if x["ok_json"]]
n = len(results)
kind = sum(1 for x in ok if x["got"]["action"].get("kind") == x["gold"]["action"].get("kind"))
mood = sum(1 for x in ok if x["got"]["mood_shift"] == x["gold"]["mood_shift"])
acted = [x for x in ok if x["gold"]["action"].get("kind") != "nenhuma"]
acted_hit = sum(1 for x in acted if x["got"]["action"].get("kind") == x["gold"]["action"].get("kind"))
false_act = sum(1 for x in ok if x["gold"]["action"].get("kind") == "nenhuma" and x["got"]["action"].get("kind") != "nenhuma")
words = [len(x["got"]["reply"].split()) for x in ok]
gold_words = [len(x["gold"]["reply"].split()) for x in results]
summary = {
    "modelo": "lora" if ADAPTER else "base",
    "exemplos": n,
    "json_valido": round(len(ok) / n, 3),
    "acao_igual": round(kind / n, 3),
    "acao_quando_devia": f"{acted_hit}/{len(acted)}",
    "acao_sem_pedir": false_act,
    "humor_igual": round(mood / n, 3),
    "palavras_media": round(sum(words) / max(len(words), 1), 1),
    "palavras_media_gabarito": round(sum(gold_words) / n, 1),
    "segundos": round(time.time() - t0),
}
print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
json.dump({"resumo": summary, "exemplos": results}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
