import argparse
import json
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_PIXELS = 768 * 768
INSTRUCTION = "Please answer with a single value."
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
REL_TOL = 0.05


def parse_number(s: str):
    t = s.strip().rstrip(".").replace(",", "")
    if t.endswith("%"):
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        return None


def normalize(s: str) -> str:
    s = s.strip().rstrip(".").lower()
    return re.sub(r"[^a-z0-9%.\- ]", "", s)


def score(gold: str, pred: str) -> bool:
    gold_num = parse_number(gold)
    if gold_num is not None:
        for m in NUM_RE.findall(pred):
            p = parse_number(m)
            if p is None:
                continue
            if gold_num == 0:
                if abs(p) <= REL_TOL:
                    return True
            elif abs(p - gold_num) / abs(gold_num) <= REL_TOL:
                return True
        return False
    gold_norm = normalize(gold)
    pred_norm = normalize(pred)
    if gold_norm == pred_norm:
        return True
    sentences = re.split(r"[.\n]", pred_norm)
    return any(gold_norm == s.strip() for s in sentences if s.strip())


def load_image(path: Path, max_pixels) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_pixels and image.width * image.height > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    return image


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ChartQA fast800 评测（relaxed accuracy）")
    ap.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-VL-4B-Instruct")
    ap.add_argument("--adapter", default="/root/autodl-tmp/saves/sft_r1")
    ap.add_argument("--base", action="store_true", help="忽略 adapter，评测基座模型")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "chartqa" / "test_fast800.json"))
    ap.add_argument("--out", default=None, help="默认 results/<adapter名>_fast800.jsonl")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="视觉像素预算，默认与 LLaMA-Factory 训练默认一致")
    ap.add_argument("--fullres", action="store_true", help="不缩放（processor 默认全分辨率，仅消融用）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    adapter = None if args.base else args.adapter
    name = Path(adapter).name if adapter else "baseline"
    max_pixels = None if args.fullres else args.max_pixels
    if max_pixels == DEFAULT_MAX_PIXELS:
        suffix = ""
    elif max_pixels is None:
        suffix = "_fullres"
    else:
        suffix = f"_px{max_pixels}"
    out_path = Path(args.out) if args.out else ROOT / "results" / f"{name}{suffix}_fast800.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = Path(args.data).parent
    items = json.loads(Path(args.data).read_text(encoding="utf-8"))

    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
    todo = [x for x in items if x["id"] not in done]
    print(f"data={args.data} out={out_path}", flush=True)
    print(f"model={args.model} adapter={adapter or '-'} max_pixels={max_pixels if max_pixels is not None else 'fullres'}", flush=True)
    print(f"total={len(items)} done={len(done)} todo={len(todo)}", flush=True)

    if todo:
        processor = AutoProcessor.from_pretrained(args.model)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype="auto", device_map="cuda")
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()

        f = out_path.open("a", encoding="utf-8")
        t0 = time.time()
        n_done = 0
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            images, texts = [], []
            for item in batch:
                images.append(load_image(base / item["image"], max_pixels))
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": str(base / item["image"])},
                    {"type": "text", "text": f"{item['question']} {INSTRUCTION}"},
                ]}]
                texts.append(processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
            inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen = out[:, inputs["input_ids"].shape[1] :]
            replies = processor.batch_decode(gen, skip_special_tokens=True)
            for item, reply in zip(batch, replies):
                reply = reply.strip()
                rec = {"id": item["id"], "type": item["type"], "gold": item["answer"], "pred": reply, "correct": score(item["answer"], reply)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(batch)
            elapsed = time.time() - t0
            print(f"{n_done}/{len(todo)} elapsed={elapsed / 60:.1f}min rate={n_done / elapsed:.2f}q/s", flush=True)
        f.close()

    recs = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
    recs = [r for r in recs if r["id"] in {x["id"] for x in items}]
    overall = sum(r["correct"] for r in recs) / len(recs)
    by_type = {}
    for tp in ("human", "augmented"):
        sub = [r for r in recs if r["type"] == tp]
        if sub:
            by_type[tp] = sum(r["correct"] for r in sub) / len(sub)
    print(f"RELAXED_ACC overall={overall:.4f} n={len(recs)} by_type={ {k: round(v, 4) for k, v in by_type.items()} }", flush=True)


if __name__ == "__main__":
    main()