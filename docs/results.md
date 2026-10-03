# 实验与结果

`posttrain-vlm` 的实验运行记录，每次跑完更新。

## 环境

| 项目 | 值 |
| --- | --- |
| GPU | 单卡 24 GB（RTX 4090 级） |
| 基座模型 | `Qwen/Qwen3-VL-4B-Instruct` |
| 训练框架 | LLaMA-Factory `TBD` |
| PyTorch / CUDA | `TBD` |
| transformers | `TBD` |

## 数据集

| 划分 | 来源 | 规模 | 用途 |
| --- | --- | --- | --- |
| 训练集 | ChartQA train | 6,000（子集，可扩充） | SFT / DPO |
| 快速验证集 | ChartQA test | 800（子集） | 快速迭代 |
| 最终评测 | ChartQA test | 全量 | 最终数字 |

**指标**：relaxed accuracy（5% 容差），定义见 [`tech-notes.md`](tech-notes.md)。

## 主结果

| # | 阶段 | ChartQA test（relaxed acc） | 备注 |
| --- | --- | --- | --- |
| 1 | Zero-shot 基线 | TBD | |
| 2 | LoRA SFT 第一轮 | TBD | |
| 3 | SFT 第二轮（数据飞轮） | TBD | |
| 4 | + DPO | TBD | |

## 消融实验

| 变量 | 取值 | 结果 | 备注 |
| --- | --- | --- | --- |
| LoRA rank | 8 / 32 | TBD | |

## 成本记录

| 运行 | GPU | 耗时 | 约成本 | 备注 |
| --- | --- | --- | --- | --- |
| 环境冒烟测试 | TBD | TBD | TBD | |
| SFT 第一轮 | TBD | TBD | TBD | |
| SFT 第二轮 | TBD | TBD | TBD | |
| DPO | TBD | TBD | TBD | |

## 失败案例

_错误分析阶段填写：代表性的错误预测、疑似原因、以及针对性补充的数据。_

## 备注 / 决策

- 每一次配置改动都要记录，并写清改动原因。
