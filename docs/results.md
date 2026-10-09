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

## 冒烟测试（合成数据，已通过）

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

> 评测口径：训练与评测统一使用视觉像素预算（评测脚本默认参数，与 LLaMA-Factory 训练默认一致）。

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
- 正式口径固化：零参数重测（训练/评测同源）得到主结果 82.25% → 83.63%。结果文件与错误分析报告已归档 `results/`。
- 决策：r1 净收益接近零的根因——格式类修复（41→9）被计算类回退（70→77）与量纲错误（1→8）抵消，训练标签存在格式歧义（见失败案例）。r2 方案：标签清洗三规则（剔除歧义小数百分比 / 数字单词规范化 / 列表仅保留多答案题）+ human 占比 2000→4000（计算类优先）。SFT 第一轮训练 9000 步 @1.42 step/s。
- 复盘教训：① 口径属于输入契约，应落在脚本默认值而非散落文档；② 标签质检前置（格式/量纲/多答案），训练前先跑统计；③ fast800 噪声约 ±1.3pp，小差异别过度解读，最终结论用全量 2500。
- r2 数据构建实测（`build_flywheel.py --stats-only` + 正式生成）：三规则命中——歧义小数百分比剔除 167（human 114 / aug 53；test 交叉检验：318 条 percent 题中 gold<1 仅 5 条 = 1.6%，drop 策略成立）；列表标签 123 留 / 48 剔（多答案线索含 and 后救回一批两问句）；数词转换 0 命中——核查 r1 train.json 含数词标签仅 2 条且为误报，此前"单词 835"实为**非数字单词答案**（Yes/No/颜色/类别等，共 1087 条），训练标签中无真数词；规则保留为保险。→ 修正前述对"数字单词规范化"的预期。
- r2 采样：human 清洗后 7236（计算类 3893），按 2:1 取 2667+1333；aug 2000；合计 6000（seed 42，`flywheel_manifest.json`）。未采用"全取计算类"——那样非数值样本只剩 115 条，有回退格式修复的风险。
- r2 数据落盘：`train_r2.json` 6000 条 + `images/train_r2/` 6000 张（238M）；`dataset_info.json` 增注册 `chartqa_sft_r2`（保留 `chartqa_sft`）。
- r2 结果归档：fast800 **87.12%**（human 80.25 / aug 94.0），全量 2500 = **85.32%**（human 76.8 / aug 93.84）；较 r1 +3.49pp（fast800），超出噪声带。主因：计算/比较类错误 77→61（human 4000 + 2:1 配比奏效），且格式修复未回退。训练 9000 步 @1.47 step/s，train_loss 0.2249；评测 6.28 q/s。产物：`results/sft_r2_fast800.jsonl`、`results/sft_r2_full.jsonl`、`results/error_analysis_r2.md`。
- DPO 执行：拒绝采样 723 对（候选=训练池中 r2 未训样本 3000 human + 1500 aug，共扫描 4500，答错才成对，防止背题污染）；超参对齐 LF 官方 `qwen3vl_lora_dpo.yaml`（β0.1 / sigmoid / lr 5e-6 / r16 续训 sft_r2 adapter；2ep×acc4 与官方 3ep×acc8 等效步数）；训练 362 步 @0.415 step/s，train_loss 0.726。结果 fast800 87.25%（+0.13）、全量 **85.84%**（+0.52，human +1.12）——不掉分、小幅正向；未大幅拉升属预期（SFT 已强 + 偏好集仅 723 + β 保守），后续可试降 β / 扩对数。产物：`results/dpo_fast800.jsonl`、`results/dpo_full.jsonl`、`results/error_analysis_dpo.md`。
- 本机部署验证（`serving/local_app.py`，RTX 4060 Ti 8GB）：单实例 + `disable_adapter()` 关/开对比（原始 vs SFT r2+DPO），云端 vLLM 留作可选。量化对照：NF4 微调侧数值题 2/5 与云端 bf16 不符 → **默认 8bit**（6/6 与云端一致；`--load-4bit` 可回退）；聊天模板一致性已验证（LF 保存模板与基座模板渲染逐字相同）。能力边界：固定 instruction 下单值回答；自由模式（不追加指令）下基座可长篇解释/推理、微调域内题仍偏单值——领域窄化演示。产物：`results/local_demo_check.md`、`docs/assets/local_demo.png`。
- vLLM 部署（单卡 4090）：`llamafactory-cli export` 合并 LoRA（8.3G；merged vs adapter fast800 87.38% vs 87.25%，23/800 文本差异为 bf16 舍入，净 +1 题）→ vLLM 0.31 独立 venv（自带 torch 2.13/CUDA13）→ OpenAI 兼容服务（mm 预算 589824 与评测同源）。压测：并发 1 平均 0.102s/9.8 req/s，并发 8 **90.4 req/s**（p50 77ms），显存 20.7G；同批 200 题 78.0% vs 云端 77.5%。产物：`results/merged_dpo_fast800.jsonl`、`serving/serve_vllm.sh`。
