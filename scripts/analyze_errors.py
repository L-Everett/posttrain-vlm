import argparse
import json
from pathlib import Path

from common import NUM_RE, parse_number

ROOT = Path(__file__).resolve().parents[1]

COMPUTE_KW = (
    "average", "mean", "median", "sum", "total", "difference", "differ",
    "how many", "increase", "decrease", "percentage of", "percent of",
    "ratio", "times", "combined", "together", "half", "double", "twice",
    "third", "quarter", "greater", "less", "higher", "lower",
)

LABELS = {
    "computation": "计算/比较类",
    "direct_read": "直接读数类",
    "scale_error": "量纲错误（×/÷100）",
    "no_number": "未输出数字",
    "format_n2s": "非数值题答成数字",
    "other_nonnumeric": "其他非数值题错误",
    "correct": "正确",
}


def extract_numbers(s):
    out = []
    for m in NUM_RE.findall(str(s)):
        v = parse_number(m)
        if v is not None:
            out.append(v)
    return out


def close(a, b):
    if b == 0:
        return abs(a) <= 0.05
    return abs(a - b) / abs(b) <= 0.05


def classify(question, gold, pred):
    q = question.lower()
    gold_num = parse_number(gold)
    pred_nums = extract_numbers(pred)
    if gold_num is None:
        return "format_n2s" if pred_nums else "other_nonnumeric"
    if pred_nums:
        if any(close(p, gold_num) for p in pred_nums):
            return "correct"
        if any(close(p * 100, gold_num) or close(p / 100, gold_num) for p in pred_nums):
            return "scale_error"
    else:
        return "no_number"
    return "computation" if any(k in q for k in COMPUTE_KW) else "direct_read"


def load_jsonl(path):
    recs = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        recs[rec["id"]] = rec
    return recs


def parse_args():
    ap = argparse.ArgumentParser(description="错误分类与对比分析（基线 vs 训练后）")
    ap.add_argument("--base", default=str(ROOT / "results" / "baseline_fast800.jsonl"))
    ap.add_argument("--pred", default=str(ROOT / "results" / "sft_r1_fast800.jsonl"))
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "chartqa" / "test_fast800.json"))
    ap.add_argument("--out", default=None, help="可选：把报告写到文件")
    ap.add_argument("--top", type=int, default=3, help="每类样例条数")
    return ap.parse_args()


def main():
    args = parse_args()
    items = {x["id"]: x for x in json.loads(Path(args.data).read_text(encoding="utf-8"))}
    base = load_jsonl(args.base)
    pred = load_jsonl(args.pred)
    ids = sorted(set(base) & set(pred) & set(items))
    pred_name = Path(args.pred).stem

    fixed = [i for i in ids if not base[i]["correct"] and pred[i]["correct"]]
    broke = [i for i in ids if base[i]["correct"] and not pred[i]["correct"]]
    base_cat = {i: classify(items[i]["question"], items[i]["answer"], base[i]["pred"]) for i in ids}
    pred_cat = {i: classify(items[i]["question"], items[i]["answer"], pred[i]["pred"]) for i in ids}
    cats = [c for c in LABELS if c != "correct"]

    lines = [
        f"# 错误分析：{Path(args.base).stem} → {pred_name}",
        "",
        f"- 题目数 {len(ids)}｜基线错误 {sum(not base[i]['correct'] for i in ids)}｜本轮错误 {sum(not pred[i]['correct'] for i in ids)}｜修好 {len(fixed)}｜弄坏 {len(broke)}",
        "",
        "## 错误类型分布",
        "",
        "| 类型 | 基线错误 | 本轮错误 | 修好 | 弄坏 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in cats:
        b_err = sum(1 for i in ids if not base[i]["correct"] and base_cat[i] == c)
        p_err = sum(1 for i in ids if not pred[i]["correct"] and pred_cat[i] == c)
        fx = sum(1 for i in fixed if base_cat[i] == c)
        bk = sum(1 for i in broke if pred_cat[i] == c)
        lines.append(f"| {LABELS[c]} | {b_err} | {p_err} | {fx} | {bk} |")

    def dump(title, group, cat_of):
        lines.append("")
        lines.append(f"## {title}")
        for c in cats:
            exs = [i for i in group if cat_of[i] == c][: args.top]
            if not exs:
                continue
            lines.append("")
            lines.append(f"**{LABELS[c]}**")
            for i in exs:
                it = items[i]
                lines.append(
                    f"- Q: {it['question'][:100]} | gold: {it['answer']} | base: {str(base[i]['pred'])[:40]} -> pred: {str(pred[i]['pred'])[:40]}"
                )

    dump("弄坏（原来对、现在错）", broke, pred_cat)
    dump("修好（原来错、现在对）", fixed, base_cat)

    report = "\n".join(lines)
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()