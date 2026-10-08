# 进度交接（内部文档 · 发布前清理）

> 新会话先读这份 + `docs/results.md`，即可无缝续做。

## 当前坐标

- 8 步清单：① ✅ → ② ✅ → ③ ✅ → ④ SFT r1 ✅ → ⑤ 错误分析 ✅ → **⑥ 数据飞轮（r2）：数据已就绪，待训练**
- 主口径结果（默认视觉预算，训练/评测同源，test_fast800）：基线 **82.25%**（human 71.75 / aug 92.75）→ r1 **83.63%**（human 73.0 / aug 94.25）；全分辨率消融见 results.md
- ⑤ 结论：格式类错误 41→9（指令跟随修好）；计算类 71→77、量纲 ×100 错误 1→8（训练标签问题），解析见 results.md
- ⑥ 状态（2026-10-08）：脚本与数据就绪，只差训练
  1. ✅ `scripts/build_flywheel.py` + `configs/sft_lora_r2.yaml`；两个待定决策已定：小数百分比**剔除**（test 仅 1.6% 小数金标）/ 数词全量解析（实测 0 命中，no-op 保险）/ 列表多答案线索含 and（123 留 48 剔）
  2. ✅ train_r2 已生成并注册：human 4000（计算类 2667，2:1）+ aug 2000；图 6000 张 238M；`flywheel_manifest.json` 锁 seed 42
  3. ⬜ 训练：`sh train.sh --config configs/sft_lora_r2.yaml`（tmux 里跑）；评测：`sh test.sh --adapter /root/autodl-tmp/saves/sft_r2`（默认参数即统一预算，无需手传）
  4. ⬜ 数字进 results.md 主表第 3 行
- 归档（2026-10-08 已完成）：默认口径重测固化正式数字；`results/error_analysis_r1.md` 已生成；本地 `results/` + 远程数据盘双备份；LF 默认值 = 768×768 已核实（与脚本默认一致）

## 环境事实

- AutoDL 北京B区 RTX 4090（实例 `autodl-container-61b54eb4f4`），SSH：`ssh -p 55212 root@connect.bjb1.seetacloud.com`（该实例端口未变过；如控制台显示不同以控制台为准）
- 镜像：PyTorch 2.8.0+cu128 / Python 3.12 / conda **base**（无独立环境）；transformers 5.6.0 + LLaMA-Factory 0.9.5 + peft 0.18.1 + trl 0.24.0 + datasets 4.0.0（版本以 results.md 环境表为准）
- 关键路径（全在数据盘，重启不丢）：
  - 模型：`/root/autodl-tmp/models/Qwen3-VL-4B-Instruct`（8.3G）
  - 仓库：`/root/posttrain-vlm`
  - 数据：`/root/posttrain-vlm/data/processed/chartqa/`（train.json 6000 条 / test_fast800.json / test_full.json 2500 / images 8500 张 / manifest.json 锁 seed=42）
  - 训练/评测日志：仓库 `logs/`（gitignore）；HF 缓存在 `/root/autodl-tmp`
- 数据池实况：train 池 human 7398 / aug 20901，取了 2000+4000；test 池 human/aug 各 1250

## 操作手册（本仓库与远程实例的协作约定）

- 免密 SSH：密钥在本机 `~/.ssh/id_ed25519`（AutoDL 改 root 密码不影响密钥通道）
- 非交互 ssh **不加载 .bashrc**：仓库内 `train.sh` / `test.sh` 已内置 PATH 与 HF 环境变量，直接跑即可；手敲 python 时先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`
- 长任务：训练在 `tmux new -s train` 会话里跑 `sh train.sh`（前台实时看 + `logs/` 落盘）；其他脚本可用 `setsid python xxx </dev/null >/root/autodl-tmp/xxx.log 2>&1 &` 脱离会话，之后轮询日志
- git 单向流：**所有提交走本机**（commit + push，需本机 Clash 代理开）→ 远程 `git pull`；远程实例没有 GitHub 凭证，不要在远程 commit 大改动（冒烟期 scp 传文件的历史已被 reset 清理，保持"本地唯一真源"）
- 远程一切 git/HF 操作前 `source /etc/network_turbo`，用完 `unset http_proxy https_proxy`（挂着会导致 pip 变慢）
- 远程 `pkill -f <模式>` 会匹配到 ssh 自身命令行导致会话自杀：模式写成 `'[p]ython xxx'` 括号形式，或复杂命令一律 scp 脚本文件上去跑（PowerShell 嵌套引号不可靠）

## 已知坑（完整记录在 docs/results.md 决策区）

1. 装/升级 LLaMA-Factory 后必查 `pip list | grep torchaudio`，必须 2.8.0+cu128
2. LF 0.9.5 视觉像素上限参数是 `image_max_pixels`（旧 `image_resolution` 已删）
3. 批量生成必须左 padding，否则分数静默变烂
4. HF 数据集下载：单文件阶段慢属正常，多 parquet 并行后速度会跃升；别中途乱切通道重跑
5. 视觉像素预算是"输入契约"：训练侧显式预算与评测口径不一致会静默低估分数（实测约 0.5pp）；已收进 `run_test.py` 默认参数，训练/评测/部署同源
