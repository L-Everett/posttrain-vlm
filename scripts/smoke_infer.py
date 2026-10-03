import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"
CASES = ROOT / "data" / "smoke" / "infer_cases.json"

processor = AutoProcessor.from_pretrained(MODEL)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype="auto", device_map="cuda")
print(f"load_time={time.time() - t0:.1f}s", flush=True)

cases = json.loads(CASES.read_text(encoding="utf-8"))
correct = 0
for idx, case in enumerate(cases):
    image = Image.open(case["image"]).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": case["image"]},
        {"type": "text", "text": case["question"]},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device, torch.bfloat16)
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    ok = reply == case["answer"]
    if not ok:
        nums = re.findall(r"-?\d+(?:\.\d+)?", reply)
        ok = bool(nums) and nums[0] == case["answer"]
    correct += int(ok)
    print(f"case{idx} expected={case['answer']!r} reply={reply!r} ok={ok} time={time.time() - t1:.1f}s", flush=True)

if torch.cuda.is_available():
    print(f"max_mem={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")
print(f"SMOKE_INFER_RESULT correct={correct}/{len(cases)}")
