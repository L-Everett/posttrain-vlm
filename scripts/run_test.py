import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import DEFAULT_MAX_PIXELS, INSTRUCTION, load_image, score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.environ.get("MODEL_PATH", "path/to/your/Qwen3-VL-4B-Instruct")
DEFAULT_ADAPTER = os.environ.get("ADAPTER_PATH", "path/to/your/saves/sft_r1")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ChartQA fast800 评测（relaxed accuracy）")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--base", action="store_true", help="忽略 adapter，评测基座模型")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "chartqa" / "test_fast800.json"))
    ap.add_argument("--out", default=None, help="默认 results/<adapter名>_fast800.jsonl")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="视觉像素预算，默认与 LLaMA-Factory 训练默认一致")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    adapter = None if args.base else args.adapter
    name = Path(adapter).name if adapter else "baseline"
    max_pixels = args.max_pixels
    suffix = "" if max_pixels == DEFAULT_MAX_PIXELS else f"_px{max_pixels}"
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
    print(f"model={args.model} adapter={adapter or '-'} max_pixels={max_pixels}", flush=True)
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
                    {"type": "image"},
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