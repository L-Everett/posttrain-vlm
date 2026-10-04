# 进度交接（内部文档 · 发布前清理）

> 新会话先读这份 + `docs/results.md`，即可无缝续做。

## 当前坐标

- 8 步清单：① 环境冒烟 ✅ → ② 数据准备 ✅ → ③ zero-shot 基线 ✅ → **下一步 ④ LoRA SFT 第一轮**
- 基线数字：overall **82.6%** / human **71.75%** / augmented **93.5%**（test_fast800，seed 42，明细在 `results/baseline_fast800.jsonl`）
- ④ 待办拆解：
  1. 写 `configs/sft_lora.yaml`（参照 `configs/smoke_lora.yaml`：`dataset_dir` 指 `data/processed/chartqa`，dataset 名 `chartqa_sft`，template `qwen3_vl_nothink`，`image_max_pixels: 602112`，rank 16 / alpha 32，3 epoch 起）
  2. 远程 `llamafactory-cli train`，预计 1~1.5h（冒烟实测 1.42 step/s @ batch1×acc2；正式 batch 2 会更快），输出存 `/root/autodl-tmp/saves/sft_r1`
  3. 评测 adapter：`eval/run_baseline.py` 加 `--adapter` 参数（`PeftModel.from_pretrained` 加载）或合并后评，用同一份 fast800 对比基线
  4. 数字进 `docs/results.md` 主结果表第 2 行

## 环境事实

- AutoDL 北京B区 RTX 4090（实例 `autodl-container-61b54eb4f4`），SSH：`ssh -p <端口> root@connect.bjb1.seetacloud.com`，**端口随重启变化**，以控制台"SSH 登录指令"为准
- 镜像：PyTorch 2.8.0+cu128 / Python 3.12 / conda **base**（无独立环境）；transformers 5.6.0 + LLaMA-Factory 0.9.5 + peft 0.18.1 + trl 0.24.0 + datasets 4.0.0（版本以 results.md 环境表为准）
- 关键路径（全在数据盘，重启不丢）：
  - 模型：`/root/autodl-tmp/models/Qwen3-VL-4B-Instruct`（8.3G）
  - 仓库：`/root/posttrain-vlm`
  - 数据：`/root/posttrain-vlm/data/processed/chartqa/`（train.json 6000 条 / test_fast800.json / test_full.json 2500 / images 8500 张 / manifest.json 锁 seed=42）
  - 训练/评测日志：`/root/autodl-tmp/*.log`；HF 缓存在 `/root/autodl-tmp`（HF_HOME 等已写 `~/.bashrc`）
- 数据池实况：train 池 human 7398 / aug 20901，取了 2000+4000；test 池 human/aug 各 1250

## 操作手册（本仓库与远程实例的协作约定）

- 免密 SSH：密钥在本机 `~/.ssh/id_ed25519`（AutoDL 改 root 密码不影响密钥通道）
- 非交互 ssh **不加载 .bashrc**：跑 Python 前先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`
- 长任务：`setsid python xxx </dev/null >/root/autodl-tmp/xxx.log 2>&1 &` 脱离会话，本地 ssh 可断，之后轮询日志
- git 单向流：**所有提交走本机**（commit + push，需本机 Clash 代理开）→ 远程 `git pull`；远程实例没有 GitHub 凭证，不要在远程 commit 大改动（冒烟期 scp 传文件的历史已被 reset 清理，保持"本地唯一真源"）
- 远程一切 git/HF 操作前 `source /etc/network_turbo`，用完 `unset http_proxy https_proxy`（挂着会导致 pip 变慢）
- 远程 `pkill -f <模式>` 会匹配到 ssh 自身命令行导致会话自杀：模式写成 `'[p]ython xxx'` 括号形式，或复杂命令一律 scp 脚本文件上去跑（PowerShell 嵌套引号不可靠）

## 已知坑（完整记录在 docs/results.md 决策区）

1. 装/升级 LLaMA-Factory 后必查 `pip list | grep torchaudio`，必须 2.8.0+cu128
2. LF 0.9.5 视觉像素上限参数是 `image_max_pixels`（旧 `image_resolution` 已删）
3. 批量生成必须左 padding，否则分数静默变烂
4. HF 数据集下载：单文件阶段慢属正常，多 parquet 并行后速度会跃升；别中途乱切通道重跑
