# 实验与结果

`posttrain-vlm` 的实验运行记录，每次跑完更新。

## 环境

| 项目 | 值 |
| --- | --- |
| GPU | 单卡 RTX 4090 24GB（AutoDL 北京B区，驱动 580.105.08） |
| 基座模型 | `Qwen/Qwen3-VL-4B-Instruct`（modelscope 下载，8.3G，位于数据盘） |
| 训练框架 | LLaMA-Factory 0.9.5 |
| PyTorch / CUDA | 2.8.0+cu128 / CUDA 12.8 |
| transformers | 5.6.0（peft 0.18.1 / trl 0.24.0 / datasets 4.0.0） |
| template | `qwen3_vl_nothink` |

## 数据集

| 划分 | 来源 | 规模 | 用途 |
| --- | --- | --- | --- |
| 训练集 | ChartQA train | 6,000（子集，可扩充） | SFT / DPO |
| 快速验证集 | ChartQA test | 800（子集） | 快速迭代 |
| 最终评测 | ChartQA test | 全量 | 最终数字 |

**指标**：relaxed accuracy（5% 容差），定义见 [`tech-notes.md`](tech-notes.md)。

## 冒烟测试（2026-10-03，合成数据，已通过）

| 项 | 结果 |
| --- | --- |
| 推理：5 张合成柱状图读数 | 5/5 正确；模型加载 3s，单题 ~3s，显存峰值 8.32GB |
| 训练：60 样本 × 5 epoch = 150 step LoRA(r8) | exit 0；loss 0.24（均值）→ ~1e-5；105s，1.42 step/s；adapter 64MB |

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
| 环境搭建+冒烟 | 4090 × 1 | ~40 min | ~1.5 元 | 修了 2 个坑（见决策） |
| SFT 第一轮 | TBD | TBD | TBD | |
| SFT 第二轮 | TBD | TBD | TBD | |
| DPO | TBD | TBD | TBD | |

## 失败案例

_错误分析阶段填写：代表性的错误预测、疑似原因、以及针对性补充的数据。_

## 备注 / 决策

- 每一次配置改动都要记录，并写清改动原因。
- [2026-10-03] 坑1：LLaMA-Factory `[torch]` extra 会把 torchaudio 拉到 2.11（CUDA13 构建），与镜像的 torch 2.8.0+cu128 冲突，报 `libcudart.so.13` 缺失 → 锁 `torchaudio==2.8.0`。以后每次装/升级 llamafactory 后都要检查 `pip list | grep torchaudio`。
- [2026-10-03] 坑2：LF 0.9.5 移除了 `image_resolution` 参数，视觉输入上限改用 `image_max_pixels` / `image_min_pixels`（在 model_args，直接放 yaml 顶层即可）。
- [2026-10-03] 坑3：非交互 SSH 不加载 .bashrc，远程跑命令要先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`。
- [2026-10-03] 长任务用 `nohup` 脱离 SSH 会话跑 + 轮询日志，避免本地超时掐断训练。
