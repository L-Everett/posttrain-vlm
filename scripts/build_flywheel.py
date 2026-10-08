import argparse
import io
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

from datasets import load_dataset
from PIL import Image

from analyze_errors import COMPUTE_KW, parse_number

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Please answer with a single value."
PERCENT_RE = re.compile(r"percent|percentage|%", re.IGNORECASE)
LIST_RE = re.compile(r"^\[.*\]$", re.DOTALL)

ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}

NUMBER_WORD_RE = re.compile(r"\b(?:" + "|".join(sorted({**ONES, **TENS, **SCALES}, key=len, reverse=True)) + r")\b")
COMPUTE_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in COMPUTE_KW) + r")\b")
MULTI_KW = ("two", "three", "four", "both", "values", "names", "pair", "each", "all", "list", "and")
MULTI_RE = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in MULTI_KW) + r")\b")

TAGS = ("percent_decimal_drop", "percent_decimal_normalize", "percent_decimal_sign_keep", "list_drop", "list_keep", "word_convert")
TAG_LABEL = {
    "percent_decimal_drop": "歧义小数百分比（剔除）",
    "percent_decimal_normalize": "歧义小数百分比（×100 归一化）",
    "percent_decimal_sign_keep": "带 % 号的小数百分比（保留）",
    "list_drop": "单答案题列表标签（剔除）",
    "list_keep": "多答案题列表标签（保留）",
    "word_convert": "单词数词（转阿拉伯数字）",
}


def parse_cardinal(tokens):
    total = 0
    current = 0
    for tok in tokens:
        if tok in ONES:
            current += ONES[tok]
        elif tok in TENS:
            current += TENS[tok]
        elif tok == "hundred":
            current = (current or 1) * SCALES[tok]
        elif tok in SCALES:
            total += (current or 1) * SCALES[tok]
            current = 0
        else:
            return None
    return total + current


def words_to_number(text):
    t = str(text).strip().lower().rstrip(".")
    tokens = [tok for tok in t.replace(",", "").replace("-", " ").split() if tok != "and"]
    if not tokens:
        return None
    if "point" in tokens:
        cut = tokens.index("point")
        head, tail = tokens[:cut], tokens[cut + 1 :]
        if not tail or any(tok not in ONES or ONES[tok] > 9 for tok in tail):
            return None
        int_part = parse_cardinal(head) if head else 0
        if int_part is None:
            return None
        return f"{int_part}." + "".join(str(ONES[tok]) for tok in tail)
    value = parse_cardinal(tokens)
    return None if value is None else str(value)


def clean_label(question, gold, percent_policy):
    g = str(gold).strip()
    q = question.lower()
    num = parse_number(g)
    if num is not None and 0 < num < 1 and PERCENT_RE.search(q):
        if "%" in g:
            return g, "percent_decimal_sign_keep"
        if percent_policy == "normalize":
            return f"{num * 100:g}", "percent_decimal_normalize"
        return None, "percent_decimal_drop"
    if LIST_RE.match(g):
        return (g, "list_keep") if MULTI_RE.search(q) else (None, "list_drop")
    converted = words_to_number(g)
    if converted is not None and converted != g:
        return converted, "word_convert"
    return g, "keep"


def clean_pool(ds, indices, percent_policy):
    kept = {}
    stats = {t: 0 for t in TAGS}
    examples = {t: [] for t in TAGS}
    for i in indices:
        item = ds[i]
        label, tag = clean_label(item["query"], item["label"], percent_policy)
        if tag in stats:
            stats[tag] += 1
            if len(examples[tag]) < 5:
                examples[tag].append(f"{item['query'][:90]} | gold={item['label']}")
        if label is not None:
            kept[i] = label
    residual = [f"{ds[i]['query'][:90]} | gold={label}" for i, label in kept.items() if NUMBER_WORD_RE.search(label)]
    return kept, stats, examples, residual


def save_image(raw, path):
    Image.open(io.BytesIO(raw)).convert("RGB").save(path)


def sft_record(image_rel, question, answer):
    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question} {INSTRUCTION}"},
            {"role": "assistant", "content": str(answer)},
        ],
        "images": [image_rel],
    }


def test_cross_check(path):
    if not path.exists():
        return None
    items = json.loads(path.read_text(encoding="utf-8"))
    total = dec = 0
    for x in items:
        if not PERCENT_RE.search(x["question"].lower()):
            continue
        total += 1
        n = parse_number(str(x["answer"]).strip())
        if n is not None and 0 < n < 1:
            dec += 1
    rate = dec / total if total else 0.0
    return f"percent 题 {total} 条，其中 gold<1 共 {dec} 条（{rate:.1%}）"


def parse_args():
    ap = argparse.ArgumentParser(description="r2 数据飞轮：标签清洗 + 补数据（human 计算类优先）")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "chartqa"))
    ap.add_argument("--train-human", type=int, default=4000)
    ap.add_argument("--train-aug", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compute-ratio", type=float, default=2.0, help="human 采样中计算类对其他的目标比例")
    ap.add_argument("--percent-policy", choices=["drop", "normalize"], default="drop")
    ap.add_argument("--stats-only", action="store_true", help="只打印清洗与抽样报告，不写任何文件")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    rng = random.Random(args.seed)

    train = load_dataset("ahmed-masry/ChartQA")["train"]
    human_idx = [i for i, t in enumerate(train["type"]) if t == "human"]
    aug_idx = [i for i, t in enumerate(train["type"]) if t == "augmented"]

    human_kept, human_stats, human_ex, human_residual = clean_pool(train, human_idx, args.percent_policy)
    aug_kept, aug_stats, aug_ex, aug_residual = clean_pool(train, aug_idx, args.percent_policy)

    need_h, need_a = args.train_human, args.train_aug
    comp, rest = [], []
    for i in human_kept:
        (comp if COMPUTE_RE.search(train[i]["query"].lower()) else rest).append(i)
    comp_set = set(comp)
    comp_need = round(need_h * args.compute_ratio / (1 + args.compute_ratio))
    picks_c = rng.sample(comp, min(comp_need, len(comp)))
    picks_r = rng.sample(rest, min(need_h - len(picks_c), len(rest)))
    short = need_h - len(picks_c) - len(picks_r)
    if short > 0:
        picked = set(picks_c) | set(picks_r)
        spare = [i for i in human_kept if i not in picked]
        picks_h = picks_c + picks_r + rng.sample(spare, min(short, len(spare)))
    else:
        picks_h = picks_c + picks_r
    n_comp = sum(1 for i in picks_h if i in comp_set)
    aug_keys = list(aug_kept)
    picks_a = rng.sample(aug_keys, min(need_a, len(aug_keys)))
    picks = sorted(picks_h + picks_a)
    labels = {**human_kept, **aug_kept}

    lines = [
        "== 数据飞轮清洗报告 ==",
        f"seed={args.seed} percent_policy={args.percent_policy} stats_only={args.stats_only}",
        f"原始池: human={len(human_idx)} aug={len(aug_idx)}",
        f"清洗后: human={len(human_kept)} aug={len(aug_kept)}",
        "规则命中:",
    ]
    for tag in TAGS:
        lines.append(f"  {TAG_LABEL[tag]}: {human_stats[tag] + aug_stats[tag]}")
        for ex in (human_ex[tag] + aug_ex[tag])[:5]:
            lines.append(f"    - {ex}")
    lines.append(f"  残余含数词标签（未转换，保留）: {len(human_residual) + len(aug_residual)}")
    for ex in (human_residual + aug_residual)[:5]:
        lines.append(f"    - {ex}")
    lines.append(f"抽样: human={len(picks_h)}（计算类 {n_comp} / 其他 {len(picks_h) - n_comp}） aug={len(picks_a)} 合计 {len(picks)}")
    lines.append(f"TEST 交叉检验: {test_cross_check(out / 'test_full.json') or 'test_full.json 不存在，跳过'}")
    print("\n".join(lines), flush=True)

    if args.stats_only:
        print("stats-only：未写任何文件", flush=True)
        return

    (out / "images" / "train_r2").mkdir(parents=True, exist_ok=True)
    records = []
    for n, i in enumerate(picks):
        item = train[i]
        rel = f"images/train_r2/{n:05d}.png"
        save_image(item["image"], out / rel)
        records.append(sft_record(rel, item["query"], labels[i]))
    (out / "train_r2.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    info_path = out / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info["chartqa_sft_r2"] = {
        "file_name": "train_r2.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset": "ahmed-masry/ChartQA",
        "seed": args.seed,
        "instruction": INSTRUCTION,
        "percent_policy": args.percent_policy,
        "compute_ratio": args.compute_ratio,
        "quotas": {"human": need_h, "aug": need_a},
        "pool": {
            "human_raw": len(human_idx),
            "aug_raw": len(aug_idx),
            "human_clean": len(human_kept),
            "aug_clean": len(aug_kept),
        },
        "cleaning": {
            "human": human_stats,
            "aug": aug_stats,
            "word_residual": {"human": len(human_residual), "aug": len(aug_residual)},
        },
        "picked": {
            "human": len(picks_h),
            "aug": len(picks_a),
            "total": len(picks),
            "human_compute": n_comp,
            "human_compute_pool": len(comp),
        },
        "picked_indices": {"human": sorted(picks_h), "aug": sorted(picks_a)},
    }
    (out / "flywheel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE train_r2.json={len(records)} images={out / 'images' / 'train_r2'}", flush=True)
    print("下一步: sh train.sh --config configs/sft_lora_r2.yaml", flush=True)


if __name__ == "__main__":
    main()