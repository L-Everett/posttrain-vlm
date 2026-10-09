import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from build_flywheel import clean_pool, save_image
from common import DEFAULT_MAX_PIXELS, INSTRUCTION, load_image, score

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description="DPO 偏好对构建（拒绝采样：答错作 rejected、金标作 chosen）")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "chartqa"))
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "path/to/your/Qwen3-VL-4B-Instruct"))
    ap.add_argument("--adapter", default=os.environ.get("ADAPTER_PATH", "path/to/your/saves/sft_r2"))
    ap.add_argument("--human", type=int, default=3000, help="human 候选数")
    ap.add_argument("--aug", type=int, default=1500, help="aug 候选数")
    ap.add_argument("--max-pairs", type=int, default=0, help="0=不设上限（冒烟可设 10）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="chartqa_dpo", help="注册的数据集名")
    ap.add_argument("--file", default="dpo_pairs.json")
    ap.add_argument("--manifest", default="flywheel_manifest.json", help="用于排除 r2 已训练样本")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="视觉像素预算，与训练/评测同源")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    rng = random.Random(args.seed)

    train = load_dataset("ahmed-masry/ChartQA")["train"]
    human_idx = [i for i, t in enumerate(train["type"]) if t == "human"]
    aug_idx = [i for i, t in enumerate(train["type"]) if t == "augmented"]
    human_kept, _, _, _ = clean_pool(train, human_idx, "drop")
    aug_kept, _, _, _ = clean_pool(train, aug_idx, "drop")
    labels = {**human_kept, **aug_kept}
    kind = {i: "human" for i in human_kept}
    kind.update({i: "aug" for i in aug_kept})

    manifest = json.loads((out / args.manifest).read_text(encoding="utf-8"))
    done = set(manifest["picked_indices"]["human"]) | set(manifest["picked_indices"]["aug"])
    cand_h = [i for i in human_kept if i not in done]
    cand_a = [i for i in aug_kept if i not in done]
    picks_h = rng.sample(cand_h, min(args.human, len(cand_h)))
    picks_a = rng.sample(cand_a, min(args.aug, len(cand_a)))
    picks = sorted(picks_h + picks_a)
    print(f"候选: human={len(picks_h)} aug={len(picks_a)} 合计 {len(picks)}（已排除 r2 已训练 {len(done)} 条）", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype="auto", device_map="cuda")
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    (out / "images" / "dpo").mkdir(parents=True, exist_ok=True)
    pairs = []
    pair_kinds = []
    created = []
    scanned = {"human": 0, "aug": 0}
    wrong = {"human": 0, "aug": 0}
    examples = []
    t0 = time.time()
    for start in range(0, len(picks), args.batch_size):
        batch = picks[start : start + args.batch_size]
        images, texts, rels = [], [], []
        for offset, i in enumerate(batch):
            item = train[i]
            rel = f"images/dpo/{start + offset:05d}.png"
            save_image(item["image"], out / rel)
            created.append(rel)
            images.append(load_image(out / rel, args.max_pixels))
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{item['query']} {INSTRUCTION}"},
            ]}]
            texts.append(processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
            rels.append(rel)
        inputs = processor(images=images, text=texts, return_tensors="pt", padding=True).to(model.device, torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        replies = processor.batch_decode(gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, item, reply, rel in zip(batch, [train[j] for j in batch], replies, rels):
            reply = reply.strip()
            scanned[kind[i]] += 1
            if not score(labels[i], reply):
                wrong[kind[i]] += 1
                pairs.append({
                    "messages": [{"role": "user", "content": f"<image>\n{item['query']} {INSTRUCTION}"}],
                    "chosen": {"role": "assistant", "content": labels[i]},
                    "rejected": {"role": "assistant", "content": reply},
                    "images": [rel],
                })
                pair_kinds.append(kind[i])
                if len(examples) < 5:
                    examples.append(f"Q: {item['query'][:90]} | gold={labels[i]} | pred={reply[:40]}")
        done_n = scanned["human"] + scanned["aug"]
        print(f"{done_n}/{len(picks)} pairs={len(pairs)} elapsed={time.time() - t0:.1f}s", flush=True)
        if args.max_pairs and len(pairs) >= args.max_pairs:
            pairs = pairs[: args.max_pairs]
            pair_kinds = pair_kinds[: args.max_pairs]
            break

    pair_counts = {"human": pair_kinds.count("human"), "aug": pair_kinds.count("aug")}

    used = {p["images"][0] for p in pairs}
    for rel in created:
        if rel not in used:
            (out / rel).unlink(missing_ok=True)

    (out / args.file).write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    info_path = out / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info[args.name] = {
        "file_name": args.file,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    dpo_manifest = {
        "model": args.model,
        "adapter": args.adapter,
        "seed": args.seed,
        "max_pixels": args.max_pixels,
        "excluded_r2_samples": len(done),
        "candidates": {"human": len(picks_h), "aug": len(picks_a)},
        "scanned": scanned,
        "wrong": wrong,
        "pairs": {"total": len(pairs), "human": pair_counts["human"], "aug": pair_counts["aug"]},
    }
    (out / f"dpo_manifest_{args.name}.json").write_text(json.dumps(dpo_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== DPO 偏好对报告 ===")
    print(f"扫描: human={scanned['human']} aug={scanned['aug']}｜答错率: human={wrong['human']}/{scanned['human']} aug={wrong['aug']}/{scanned['aug']}")
    print(f"偏好对: {len(pairs)}（human {pair_counts['human']} / aug {pair_counts['aug']}）")
    print("样例:")
    for ex in examples:
        print(f"  - {ex}")
    print(f"DONE {args.file}={len(pairs)} images={out / 'images' / 'dpo'} 注册={args.name}")


if __name__ == "__main__":
    main()