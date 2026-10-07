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
| 训练集 | ChartQA train（池：human 7398 / aug 20901） | 6,000（2000 human + 4000 aug，seed 42，见 manifest.json） | SFT / DPO |
| 快速验证集 | ChartQA test（池 2500：human/aug 各半） | 800（分层 400+400，seed 42） | 快速迭代 |
| 最终评测 | ChartQA test | 全量 2500 | 最终数字 |

**指标**：relaxed accuracy（5% 容差），定义见 [`tech-notes.md`](tech-notes.md)。

## 冒烟测试（2026-10-03，合成数据，已通过）

| 项 | 结果 |
| --- | --- |
| 推理：5 张合成柱状图读数 | 5/5 正确；模型加载 3s，单题 ~3s，显存峰值 8.32GB |
| 训练：60 样本 × 5 epoch = 150 step LoRA(r8) | exit 0；loss 0.24（均值）→ ~1e-5；105s，1.42 step/s；adapter 64MB |

## 主结果

| # | 阶段 | ChartQA test（relaxed acc） | 备注 |
| --- | --- | --- | --- |
| 1 | Zero-shot 基线 | **82.6%**（fast800） | human 72.0% / aug 93.25%；原始模型裸跑，human 题型是提升空间所在 |
| 2 | LoRA SFT 第一轮 | **83.75%**（fast800） | human 73.25% / aug 94.25%；较基线 +1.1pp；格式类错误大幅修复、计算类回退（见失败案例） |
| 3 | SFT 第二轮（数据飞轮） | TBD | |
| 4 | + DPO | TBD | |

> 评测口径：训练与评测统一使用视觉像素预算（评测脚本默认参数，与 LLaMA-Factory 训练默认一致）；"全分辨率"仅作消融。

## 消融实验

| 变量 | 取值 | 结果 | 备注 |
| --- | --- | --- | --- |
| LoRA rank | 8 / 32 | TBD | |
| 视觉预算 | 统一预算 / 全分辨率 | 83.75 / 83.13（SFT r1） | 全分辨率下分数被低估约 0.6pp；口径不一致会静默伤分 |

## 成本记录

| 运行 | GPU | 耗时 | 约成本 | 备注 |
| --- | --- | --- | --- | --- |
| 环境搭建+冒烟 | 4090 × 1 | ~40 min | ~1.5 元 | 修了 2 个坑（见决策） |
| ChartQA 下载+数据准备 | 4090 × 1 | ~20 min | ~0.7 元 | 实际下载仅用 2.49G |
| Zero-shot 基线（800 题） | 4090 × 1 | 1.8 min | ~0.07 元 | 批量推理摊薄后极快 |
| SFT 第一轮 | 4090 × 1 | 1h45m | ~3.5 元 | 9000 步 @1.42 step/s，与冒烟预测一致 |
| 口径对照评测（2 次） | 4090 × 1 | ~5 min | ~0.1 元 | 统一预算 vs 全分辨率 |
| SFT 第二轮 | TBD | TBD | TBD | |
| DPO | TBD | TBD | TBD | |

## 失败案例

- 格式类修复明显：非数值题答成数字 41 → 9。例：`Is the percentage value of "STEM" segment 52?`（gold Yes；基座输出 52 → SFT 输出 Yes）
- 计算/比较类回退：70 → 77（弄坏 21 / 修好 13）。例：`What is the difference between the highest percentage and lowest percentage?`（gold 61；基座 64 → SFT 53）
- 量纲错误（×/100）：1 → 8。例：`What percent who think of ... Dangerous?`（gold 62；基座 62 → SFT 0.62）
- 训练集标签格式统计（6000 条）：小数百分比 180 / 列表 50 / 单词 835 —— r2 先做标签治理
- 分类报告：`results/error_analysis_r1.md`

## 备注 / 决策

- 每一次配置改动都要记录，并写清改动原因。
- [2026-10-03] 坑1：LLaMA-Factory `[torch]` extra 会把 torchaudio 拉到 2.11（CUDA13 构建），与镜像的 torch 2.8.0+cu128 冲突，报 `libcudart.so.13` 缺失 → 锁 `torchaudio==2.8.0`。以后每次装/升级 llamafactory 后都要检查 `pip list | grep torchaudio`。
- [2026-10-03] 坑2：LF 0.9.5 移除了 `image_resolution` 参数，视觉输入上限改用 `image_max_pixels` / `image_min_pixels`（在 model_args，直接放 yaml 顶层即可）。
- [2026-10-03] 坑3：非交互 SSH 不加载 .bashrc，远程跑命令要先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`。
- [2026-10-03] 长任务用 `nohup` 脱离 SSH 会话跑 + 轮询日志，避免本地超时掐断训练。
- [2026-10-04] 坑4：批量生成必须 `tokenizer.padding_side="left"`，右 padding 会静默污染 batch 内短序列的生成（跑完不报错、只是分数烂）。评测类脚本统一左 padding。
- [2026-10-03] 远程 git 直连 GitHub 会 TLS 中断，fetch/push 前 `source /etc/network_turbo`（AutoDL 学术加速，用完 `unset http_proxy https_proxy`，否则 pip 会变慢）。
- [2026-10-07] 坑5：训练/评测像素口径不一致（训练侧设了视觉预算，评测侧用 processor 默认全分辨率）→ 800 题中约 5% 的大图受影响，SFT 分数被低估约 0.6pp。规则：像素预算属于输入契约，训练/评测/部署必须同值；预算已收进 `run_test.py` 默认参数（与 LLaMA-Factory 训练默认一致），不在配置/文档里写死。
- [2026-10-07] 决策：r1 净收益接近零的根因——格式类修复（41→9）被计算类回退（70→77）与量纲错误（1→8）抵消，训练标签存在格式歧义（见失败案例）。r2 方案：标签清洗三规则（剔除歧义小数百分比 / 数字单词规范化 / 列表仅保留多答案题）+ human 占比 2000→4000（计算类优先）。
- [2026-10-07] 复盘教训：① 口径属于输入契约，应落在脚本默认值而非散落文档；② 标签质检前置（格式/量纲/多答案），训练前先跑统计；③ fast800 噪声约 ±1.3pp，小差异别过度解读，最终结论用全量 2500。
