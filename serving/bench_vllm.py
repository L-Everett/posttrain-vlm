import argparse
import base64
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "chartqa"
INSTRUCTION = "Please answer with a single value."
sys.path.insert(0, str(ROOT / "scripts"))
from run_test import score  # noqa: E402


def ask(base_url, item, max_tokens=64):
    img_b64 = base64.b64encode((DATA / item["image"]).read_bytes()).decode()
    payload = {
        "model": "chartqa-vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": f"{item['question']} {INSTRUCTION}"},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.time()
    r = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    dt = time.time() - t0
    return r.json()["choices"][0]["message"]["content"].strip(), dt


def run_conc(items, base_url, conc, verbose=False):
    latencies, correct, outputs = [], 0, []
    if conc == 1:
        for it in items:
            pred, dt = ask(base_url, it)
            latencies.append(dt)
            correct += score(it["answer"], pred)
            outputs.append(pred)
            if verbose:
                print(f"Q: {it['question'][:80]} | gold={it['answer']} | pred={pred} | {dt:.2f}s", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            for pred, dt in ex.map(lambda it: ask(base_url, it), items):
                latencies.append(dt)
                outputs.append(pred)
        correct = sum(score(it["answer"], p) for it, p in zip(items, outputs))
    return latencies, correct, outputs


def main():
    ap = argparse.ArgumentParser(description="vLLM OpenAI 接口冒烟/压测")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    items = json.loads((DATA / "test_fast800.json").read_text(encoding="utf-8"))[: args.n]
    print(f"items={len(items)} conc={args.conc}", flush=True)
    t0 = time.time()
    latencies, correct, _ = run_conc(items, args.base_url, args.conc, args.verbose)
    wall = time.time() - t0
    latencies_sorted = sorted(latencies)
    print(json.dumps({
        "requests": len(items),
        "concurrency": args.conc,
        "wall_s": round(wall, 2),
        "throughput_req_s": round(len(items) / wall, 2),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_p50_s": round(latencies_sorted[len(latencies_sorted) // 2], 3),
        "latency_p95_s": round(latencies_sorted[min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95))], 3),
        "relaxed_acc": round(correct / len(items), 4),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()