import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "smoke"
IMAGES = OUT / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

CATS = ["Q1", "Q2", "Q3", "Q4"]

train_samples = []
infer_cases = []

for i in range(20):
    values = [random.randint(10, 90) for _ in CATS]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
    bars = ax.bar(CATS, values)
    ax.bar_label(bars, fmt="%d")
    ax.set_title("Sales by quarter (unit: k)")
    ax.set_ylabel("Sales")
    img_path = IMAGES / f"chart_{i:03d}.png"
    fig.savefig(img_path)
    plt.close(fig)

    j = random.randrange(len(CATS))
    questions = [(f"What is the value of {CATS[j]}?", str(values[j]))]
    questions.append(("Which quarter has the highest sales?", CATS[values.index(max(values))]))
    k = random.choice([x for x in range(len(CATS)) if x != j])
    questions.append((f"What is the value of {CATS[k]}?", str(values[k])))

    for q, a in questions:
        train_samples.append({
            "messages": [
                {"role": "user", "content": f"<image>\n{q} Answer with the value only."},
                {"role": "assistant", "content": a},
            ],
            "images": [f"images/{img_path.name}"],
        })
    if i < 5:
        infer_cases.append({"image": str(img_path), "question": questions[0][0], "answer": questions[0][1]})

(OUT / "train.json").write_text(json.dumps(train_samples, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "infer_cases.json").write_text(json.dumps(infer_cases, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "dataset_info.json").write_text(json.dumps({
    "smoke_chart": {
        "file_name": "train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }
}, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"images={len(list(IMAGES.glob('*.png')))} samples={len(train_samples)} infer_cases={len(infer_cases)}")
