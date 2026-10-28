import json, os, sys, time, re
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "Qwen/Qwen3-8B"
ADAPTER = os.environ.get("ADAPTER", "")
PORT = int(os.environ.get("PORT", "8080"))

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
model.eval()
lock = Lock()
print("pronto", "adaptador" if ADAPTER else "base", flush=True)


def generate(body):
    prompt = tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True, **body.get("chat_template_kwargs", {"enable_thinking": False}))
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(0)
    temp = float(body.get("temperature", 0.7))
    if "seed" in body:
        torch.manual_seed(int(body["seed"]) & 0x7FFFFFFF)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=min(int(body.get("max_tokens", 400)), 400), do_sample=temp > 0, temperature=max(temp, 1e-5), top_p=float(body.get("top_p", 0.9)), pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new = out[0][ids["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    dt = time.time() - t0
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": int(ids["input_ids"].shape[1]), "completion_tokens": int(len(new))}, "timings": {"predicted_per_second": len(new) / dt if dt > 0 else 0.0}}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        with lock:
            res = generate(body)
        data = json.dumps(res, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
