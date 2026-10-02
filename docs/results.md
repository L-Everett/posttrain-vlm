# Experiments & Results

Running log for `posttrain-vlm`. Update after every run.

## Environment

| Item | Value |
| --- | --- |
| GPU | Single 24 GB (RTX 4090 class) |
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Training framework | LLaMA-Factory `TBD` |
| PyTorch / CUDA | `TBD` |
| transformers | `TBD` |

## Dataset

| Split | Source | Size | Purpose |
| --- | --- | --- | --- |
| Train | ChartQA train | 6,000 (subset, expandable) | SFT / DPO |
| Held-out | ChartQA test | 800 (subset) | fast iteration |
| Final eval | ChartQA test | full | final numbers |

**Metric**: relaxed accuracy (5% tolerance) — definition in [`tech-notes.md`](tech-notes.md).

## Main results

| # | Stage | ChartQA test (relaxed acc) | Notes |
| --- | --- | --- | --- |
| 1 | Zero-shot baseline | TBD | |
| 2 | LoRA SFT round 1 | TBD | |
| 3 | SFT round 2 (data flywheel) | TBD | |
| 4 | + DPO | TBD | |

## Ablation

| Variable | Values | Result | Notes |
| --- | --- | --- | --- |
| LoRA rank | 8 / 32 | TBD | |

## Cost log

| Run | GPU | Wall time | Approx. cost | Notes |
| --- | --- | --- | --- | --- |
| Environment smoke test | TBD | TBD | TBD | |
| SFT round 1 | TBD | TBD | TBD | |
| SFT round 2 | TBD | TBD | TBD | |
| DPO | TBD | TBD | TBD | |

## Failure cases

_To be filled during error analysis: representative wrong predictions, suspected causes, and the targeted data added in response._

## Notes / decisions

- Record every config change and the reason behind it.