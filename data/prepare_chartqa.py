import argparse
import io
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Please answer with a single value."


def save_image(raw: bytes, path: Path) -> None:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.save(path)


def sft_record(image_rel: str, question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question} {INSTRUCTION}"},
            {"role": "assistant", "content": str(answer)},
        ],
        "images": [image_rel],
    }


def eval_record(rid: int, item: dict, image_rel: str) -> dict:
    return {
        "id": rid,
        "type": item["type"],
        "question": item["query"],
        "answer": str(item["label"]),
        "image": image_rel,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "chartqa"))
    ap.add_argument("--train-human", type=int, default=2000)
    ap.add_argument("--train-aug", type=int, default=4000)
    ap.add_argument("--fast-human", type=int, default=400)
    ap.add_argument("--fast-aug", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "test").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    ds = load_dataset("ahmed-masry/ChartQA")
    train, test = ds["train"], ds["test"]

    train_human = [i for i, t in enumerate(train["type"]) if t == "human"]
    train_aug = [i for i, t in enumerate(train["type"]) if t == "augmented"]
    print(f"train pool: human={len(train_human)} aug={len(train_aug)}")
    picks = sorted(
        rng.sample(train_human, min(args.train_human, len(train_human)))
        + rng.sample(train_aug, min(args.train_aug, len(train_aug)))
    )

    sft_records = []
    for n, i in enumerate(picks):
        item = train[i]
        rel = f"images/train/{n:05d}.png"
        save_image(item["image"], out / rel)
        sft_records.append(sft_record(rel, item["query"], item["label"]))

    test_human = [i for i in range(len(test)) if test[i]["type"] == "human"]
    test_aug = [i for i in range(len(test)) if test[i]["type"] == "augmented"]
    print(f"test pool: human={len(test_human)} aug={len(test_aug)}")

    test_records = []
    for i in range(len(test)):
        item = test[i]
        rel = f"images/test/{i:04d}.png"
        save_image(item["image"], out / rel)
        test_records.append(eval_record(i, item, rel))

    fast_ids = sorted(
        rng.sample(test_human, min(args.fast_human, len(test_human)))
        + rng.sample(test_aug, min(args.fast_aug, len(test_aug)))
    )
    fast_records = [test_records[i] for i in fast_ids]

    (out / "train.json").write_text(json.dumps(sft_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "test_full.json").write_text(json.dumps(test_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "test_fast800.json").write_text(json.dumps(fast_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dataset_info.json").write_text(
        json.dumps(
            {
                "chartqa_sft": {
                    "file_name": "train.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages", "images": "images"},
                    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "dataset": "ahmed-masry/ChartQA",
        "seed": args.seed,
        "instruction": INSTRUCTION,
        "train": {"human": len(train_human), "aug": len(train_aug), "picked": len(picks)},
        "test": {"full": len(test_records), "fast": len(fast_records), "fast_ids": fast_ids},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", json.dumps(manifest["train"]), "fast:", len(fast_records))


if __name__ == "__main__":
    main()
