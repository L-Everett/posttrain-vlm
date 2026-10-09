# posttrain-vlm

**可复现的视觉语言模型（VLM）领域后训练全流程 —— LoRA SFT、数据飞轮、DPO 偏好对齐，以 Qwen3-VL-4B + ChartQA 为例完整演示。**

## 项目简介

通用 VLM 的 zero-shot 能力已经很强，但针对特定领域做后训练，通常还能再涨一大截。本项目搭一条小而全、完全可复现的流程，把 `Qwen/Qwen3-VL-4B-Instruct` 适配到**图表问答**（ChartQA）任务，并在留出的测试集上对每个阶段做量化评测：

1. **LoRA SFT** —— 用领域指令数据做监督微调
2. **数据飞轮** —— 对 SFT 模型做错误分析 → 针对性补数据 → 第二轮 SFT
3. **DPO** —— 用拒绝采样从 SFT 模型构造偏好对，做 Direct Preference Optimization 对齐
4. **部署** —— LoRA 合并回基座，vLLM 起服务 + Gradio 简单 demo

## 结果

ChartQA 测试集上的 relaxed accuracy（5% 容差）。详细表格和失败案例分析见 [`docs/results.md`](docs/results.md)。

| 阶段 | Relaxed Accuracy |
| --- | --- |
| Qwen3-VL-4B-Instruct（zero-shot） | 82.25% |
| + LoRA SFT | 83.63% |
| + 数据飞轮（SFT 第二轮） | 87.12%（fast800）/ 85.32%（全量） |
| + DPO | 87.25%（fast800）/ 85.84%（全量） |

> fast800 快速验证集、统一评测口径。

## 快速开始

把配置/脚本里的 `path/to/your/...` 替换成你自己的路径（基座模型、输出目录等）；数据与结果的默认目录都在仓库内。所有命令都在仓库根目录执行。

### 0. 环境

```bash
conda create -n ai python=3.10 -y && conda activate ai
pip install -r requirements-train.txt            # 训练 / 评测
pip install -r serving/requirements-local.txt    # 可选：本机 demo
```

### 1. 准备基座模型

从 ModelScope 下载 `Qwen/Qwen3-VL-4B-Instruct` 到 `path/to/your/Qwen3-VL-4B-Instruct`。本项目不分发模型权重；微调产物是几十到几百 MB 的 LoRA adapter，自备即可。

### 2. 准备数据

```bash
python data/prepare_chartqa.py    # 下载 ChartQA，生成 data/processed/chartqa（seed 42）
```

### 3. 冒烟自检（可选）

```bash
python scripts/smoke_data.py                     # 合成柱状图数据（已随仓库提供 data/smoke）
sh train.sh --config configs/smoke_lora.yaml     # 小样本训练
python scripts/smoke_infer.py                    # 合成图读数自检
```

### 4. SFT 第一轮

```bash
sh train.sh                                      # 默认 configs/sft_lora.yaml
sh test.sh --adapter path/to/your/saves/sft_r1   # 评测 → results/*.jsonl
```

### 5. 数据飞轮（第二轮 SFT）

```bash
python scripts/build_flywheel.py                 # 标签清洗 + 计算类优先采样 → train_r2.json
sh train.sh --config configs/sft_lora_r2.yaml
sh test.sh --adapter path/to/your/saves/sft_r2
```

### 6. DPO 对齐

```bash
python scripts/build_dpo.py                      # 拒绝采样构建偏好对 → dpo_pairs.json
sh train.sh --config configs/dpo_lora.yaml
sh test.sh --adapter path/to/your/saves/dpo
python scripts/analyze_errors.py --pred results/dpo_fast800.jsonl
```

### 7. 合并与部署

```bash
llamafactory-cli export configs/merge_lora.yaml  # LoRA 合并回基座
sh serving/serve_vllm.sh                         # vLLM OpenAI 兼容服务
python serving/bench_vllm.py --n 200 --conc 8    # 压测
```

本机 8 GB 显存即可跑的量化对比 demo（无需云 GPU）：

```bash
python serving/local_app.py          # 默认 8bit；--load-4bit 切 NF4
python serving/local_app.py --smoke  # 命令行自检
```

## 目录结构

```
posttrain-vlm/
├── train.sh   # 一键训练（前台运行，实时输出 + logs/ 落盘）
├── test.sh    # 一键评测（默认评最新 adapter）
├── configs/   # 训练配置（LoRA SFT、数据飞轮 r2、DPO、merge）
├── data/      # 数据准备脚本 + smoke 合成数据
├── scripts/   # 入口脚本与数据处理（run_train / run_test / build_flywheel / build_dpo / analyze_errors）
├── serving/   # 本机 8bit 对比 demo（Gradio）+ vLLM 服务脚本
├── results/   # 各阶段评测输出（jsonl）与错误分析报告
├── models/    # 基座模型与 adapter 放置处（gitignore，不随仓库分发）
└── docs/      # 实验记录与技术笔记
```

## 硬件

所有实验都按**单张 24 GB 显卡**（RTX 4090 级）设计。

## 本地部署（量化对比 demo）

![本地对比 demo](docs/assets/local_demo.png)

不依赖云 GPU：在 8 GB 显存的机器上以 **8bit** 加载 Qwen3-VL-4B（更接近 bf16 口径；`--load-4bit` 可切 NF4 省显存），并用 PEFT 的 `disable_adapter()` 在**同一模型实例**上关闭/开启微调增量——同一张图 + 同一问题，左右对比「原始模型 vs SFT r2 + DPO」的输出。

- 环境：conda env `ai`（Python 3.10 + torch 2.7.1+cu126）；依赖版本见 [`serving/requirements-local.txt`](serving/requirements-local.txt)
- 基座：`path/to/your/Qwen3-VL-4B-Instruct`（ModelScope 下载）；adapter：`path/to/your/dpo_adapter`
- 启动：双击 `serving/start_demo.bat`，或 `conda activate ai && python serving/local_app.py`
- 浏览器打开 `http://127.0.0.1:7860`；`python serving/local_app.py --smoke` 为命令行自检
- 推理口径与云端评测一致（同 instruction、视觉预算 768×768、左 padding、greedy）；8bit 下 6/6 测试题与云端 bf16 结果一致（见 [`results/local_demo_check.md`](results/local_demo_check.md)）
- 模型边界：这是 ChartQA 单值问答模型，解释/闲聊会被压成一个值；勾选界面上的「自由提问模式」可观察不加指令时的行为（基座会长篇解释、微调仍偏单值）

### 云端 vLLM 服务

LoRA 与基座合并（`configs/merge_lora.yaml`，8.3G）后，用 vLLM 0.31 起 OpenAI 兼容服务（`serving/serve_vllm.sh`，mm 预算与评测同源）：

| 项 | 值 |
| --- | --- |
| 单请求 | 平均 0.10s，p50 0.11s（200 题，4090） |
| 并发 8 | **90 req/s**，p50 77ms |
| 显存 | 20.7G / 24G（util 0.85） |
| 正确性 | 合并前后 87.38% vs 87.25%（±1 题）；同批题目 vLLM 78.0% vs adapter 77.5% |

## 状态

已完成并归档 —— 8 步闭环（数据 → 基线 → SFT → 错误分析 → 数据飞轮 → DPO → 合并部署）全部跑通；本机 8bit 对比 demo 与云端 vLLM 服务均已实测。

## 致谢

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) —— 统一微调框架（Apache-2.0）
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) —— 基座视觉语言模型（Apache-2.0）
- [ChartQA](https://github.com/vis-nlp/ChartQA) —— 图表问答基准（许可证见原仓库）

## License

Apache-2.0 —— 见 [LICENSE](LICENSE)。
