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
| 训练集 r2 | 同一池，标签清洗后重建（seed 42，见 flywheel_manifest.json） | 6,000（4000 human + 2000 aug） | SFT r2 |
| 偏好对 | 训练池未训样本拒绝采样（r2 答错才成对，seed 42，见 dpo_manifest_chartqa_dpo.json） | 723（human 594 + aug 129） | DPO |
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
| 1 | Zero-shot 基线 | **82.25%**（fast800） | human 71.75% / aug 92.75%；原始模型裸跑，human 题型是提升空间所在 |
| 2 | LoRA SFT 第一轮 | **83.63%**（fast800） | human 73.0% / aug 94.25%；较基线 +1.4pp；格式类错误大幅修复、计算类回退（见失败案例） |
| 3 | SFT 第二轮（数据飞轮） | **87.12%**（fast800）/ 85.32%（全量2500） | human 80.25% / aug 94.0%；较 r1 +3.49pp；计算类 77→61、量纲 8→5、格式零回退（见失败案例） |
| 4 | + DPO | **87.25%**（fast800）/ 85.84%（全量2500） | human 80.5% / aug 94.0%（fast800）；较 r2 +0.13pp（fast800）/ +0.52pp（全量）；human +1.12pp，不掉分（见失败案例） |

> 评测口径：训练与评测统一使用视觉像素预算（评测脚本默认参数，与 LLaMA-Factory 训练默认一致）；"全分辨率"仅作消融。

## 消融实验

| 变量 | 取值 | 结果 | 备注 |
| --- | --- | --- | --- |
| LoRA rank | 8 / 32 | TBD | |
| 视觉预算 | 默认（训练同源） / 全分辨率 | 83.63 / 83.13（SFT r1） | 全分辨率会低估约 0.5pp；口径不一致会静默伤分 |

## 成本记录

| 运行 | GPU | 耗时 | 约成本 | 备注 |
| --- | --- | --- | --- | --- |
| 环境搭建+冒烟 | 4090 × 1 | ~40 min | ~1.5 元 | 修了 2 个坑（见决策） |
| ChartQA 下载+数据准备 | 4090 × 1 | ~20 min | ~0.7 元 | 实际下载仅用 2.49G |
| Zero-shot 基线（800 题） | 4090 × 1 | 1.8 min | ~0.07 元 | 批量推理摊薄后极快 |
| SFT 第一轮 | 4090 × 1 | 1h45m | ~3.5 元 | 9000 步 @1.42 step/s，与冒烟预测一致 |
| 口径对照评测（2 次） | 4090 × 1 | ~5 min | ~0.1 元 | 统一预算 vs 全分辨率 |
| 默认口径重测（2 次） | 4090 × 1 | ~4 min | ~0.1 元 | 固化正式数字 |
| SFT 第二轮 | 4090 × 1 | 1h43m | ~3.5 元 | 9000 步 @1.47 step/s，train_loss 0.2249 |
| r2 评测（fast800 + 全量） | 4090 × 1 | 2.1 + 6.6 min | ~0.15 元 | 6.28 q/s |
| DPO 偏好对构建 | 4090 × 1 | 14.3 min | ~0.5 元 | 4500 候选扫描 → 723 对（错率 human 19.8% / aug 8.6%） |
| DPO 训练 | 4090 × 1 | 23.3 min | ~0.8 元 | 362 步 @0.415 step/s，train_loss 0.726 |
| DPO 评测（fast800 + 全量） | 4090 × 1 | 2.1 + 6.6 min | ~0.15 元 | |

## 失败案例

- 格式类修复明显：非数值题答成数字 41 → 9。例：`Is the percentage value of "STEM" segment 52?`（gold Yes；基座输出 52 → SFT 输出 Yes）
- 计算/比较类回退：71 → 77（弄坏 20 / 修好 13）。例：`What is the difference between the highest percentage and lowest percentage?`（gold 61；基座 64 → SFT 53）
- 量纲错误（×/100）：1 → 8。例：`What percent who think of ... Dangerous?`（gold 62；基座 62 → SFT 0.62）
- 训练集标签格式统计（6000 条）：小数百分比 180 / 列表 50 / 单词 835 —— r2 先做标签治理
- 分类报告：`results/error_analysis_r1.md`
- r2 相对 r1（fast800）：错误 131→103（修好 42 / 弄坏 14）；计算/比较类 77→61、量纲 8→5、非数值题格式 9→7（零新增弄坏）。修回案例：`How many waited in Total for 10mins?`（r1 14 → r2 33）、`What percent ... Dangerous?`（r1 0.62 → r2 62）
- 分类报告：`results/error_analysis_r2.md`
- DPO 相对 r2（fast800）：错误 103→102（修 6 / 破 5）——计算类 61→59、量纲 5→4 微降；非数值格式 7→8（2 个新回退，如 `biggest response for Democrats?` Same→33）。全量 human 76.8→77.92（+1.12pp）
- 分类报告：`results/error_analysis_dpo.md`

## 备注 / 决策

- 每一次配置改动都要记录，并写清改动原因。
- [2026-10-03] 坑1：LLaMA-Factory `[torch]` extra 会把 torchaudio 拉到 2.11（CUDA13 构建），与镜像的 torch 2.8.0+cu128 冲突，报 `libcudart.so.13` 缺失 → 锁 `torchaudio==2.8.0`。以后每次装/升级 llamafactory 后都要检查 `pip list | grep torchaudio`。
- [2026-10-03] 坑2：LF 0.9.5 移除了 `image_resolution` 参数，视觉输入上限改用 `image_max_pixels` / `image_min_pixels`（在 model_args，直接放 yaml 顶层即可）。
- [2026-10-03] 坑3：非交互 SSH 不加载 .bashrc，远程跑命令要先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`。
- [2026-10-03] 长任务用 `nohup` 脱离 SSH 会话跑 + 轮询日志，避免本地超时掐断训练。
- [2026-10-04] 坑4：批量生成必须 `tokenizer.padding_side="left"`，右 padding 会静默污染 batch 内短序列的生成（跑完不报错、只是分数烂）。评测类脚本统一左 padding。
- [2026-10-03] 远程 git 直连 GitHub 会 TLS 中断，fetch/push 前 `source /etc/network_turbo`（AutoDL 学术加速，用完 `unset http_proxy https_proxy`，否则 pip 会变慢）。
- [2026-10-07] 坑5：训练/评测像素口径不一致（训练侧设了视觉预算，评测侧用 processor 默认全分辨率）→ 800 题中约 5% 的大图受影响，SFT 分数被低估约 0.5pp。规则：像素预算属于输入契约，训练/评测/部署必须同值；预算已收进 `run_test.py` 默认参数（与 LLaMA-Factory 训练默认一致），不在配置/文档里写死。r1 训练侧为一次性略高预算（与现行默认相差约 2%），结论不受影响；r2 起训练/评测完全同源。
- [2026-10-08] 正式口径固化：零参数重测（训练/评测同源）得到主结果 82.25% → 83.63%；全分辨率对照保留为消融。结果文件与错误分析报告已归档 `results/`。
- [2026-10-07] 决策：r1 净收益接近零的根因——格式类修复（41→9）被计算类回退（70→77）与量纲错误（1→8）抵消，训练标签存在格式歧义（见失败案例）。r2 方案：标签清洗三规则（剔除歧义小数百分比 / 数字单词规范化 / 列表仅保留多答案题）+ human 占比 2000→4000（计算类优先）。
- [2026-10-07] 复盘教训：① 口径属于输入契约，应落在脚本默认值而非散落文档；② 标签质检前置（格式/量纲/多答案），训练前先跑统计；③ fast800 噪声约 ±1.3pp，小差异别过度解读，最终结论用全量 2500。
- [2026-10-08] r2 数据构建实测（`build_flywheel.py --stats-only` + 正式生成）：三规则命中——歧义小数百分比剔除 167（human 114 / aug 53；test 交叉检验：318 条 percent 题中 gold<1 仅 5 条 = 1.6%，drop 策略成立）；列表标签 123 留 / 48 剔（多答案线索含 and 后救回一批两问句）；数词转换 0 命中——核查 r1 train.json 含数词标签仅 2 条且为误报，此前"单词 835"实为**非数字单词答案**（Yes/No/颜色/类别等，共 1087 条），训练标签中无真数词；规则保留为保险。→ 修正 [2026-10-07] 中对"数字单词规范化"的预期。
- [2026-10-08] r2 采样：human 清洗后 7236（计算类 3893），按 2:1 取 2667+1333；aug 2000；合计 6000（seed 42，`flywheel_manifest.json`）。未采用"全取计算类"——那样非数值样本只剩 115 条，有回退格式修复的风险。
- [2026-10-08] r2 数据落盘：`train_r2.json` 6000 条 + `images/train_r2/` 6000 张（238M）；`dataset_info.json` 增注册 `chartqa_sft_r2`（保留 `chartqa_sft`）。
- [2026-10-08] r2 结果归档：fast800 **87.12%**（human 80.25 / aug 94.0），全量 2500 = **85.32%**（human 76.8 / aug 93.84）；较 r1 +3.49pp（fast800），超出噪声带。主因：计算/比较类错误 77→61（human 4000 + 2:1 配比奏效），且格式修复未回退。产物：`results/sft_r2_fast800.jsonl`、`results/sft_r2_full.jsonl`、`results/error_analysis_r2.md`。
- [2026-10-08] 坑6：LF CLI 覆盖参数必须写 `key=value`（`OmegaConf.from_cli`），`--key value` 会被 HfArgumentParser 拒绝并报 unused keys——`run_train.py` 的覆盖功能此前从未被实测，DPO 冒烟时修复。
- [2026-10-08] DPO 执行：拒绝采样 723 对（候选=训练池中 r2 未训样本 3000 human + 1500 aug，答错才成对，防止背题污染）；超参对齐 LF 官方 `qwen3vl_lora_dpo.yaml`（β0.1 / sigmoid / lr 5e-6 / r16 续训 sft_r2 adapter；2ep×acc4 与官方 3ep×acc8 等效步数）。结果 fast800 87.25%（+0.13）、全量 **85.84%**（+0.52，human +1.12）——不掉分、小幅正向；未大幅拉升属预期（SFT 已强 + 偏好集仅 723 + β 保守），后续可试降 β / 扩对数。产物：`results/dpo_fast800.jsonl`、`results/dpo_full.jsonl`、`results/error_analysis_dpo.md`。
