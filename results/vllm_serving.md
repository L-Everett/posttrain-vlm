# vLLM 部署验证（2026-10-08）

在单卡 4090 24GB 上把 SFT r2 + DPO 的 LoRA 与基座合并，用 vLLM 0.31.0 起 OpenAI 兼容服务并做小压测。

## 环境与启动

- vLLM **0.31.0**，独立 venv（`/root/autodl-tmp/venv-vllm`，自带 torch 2.13 + CUDA 13 全栈；不污染训练环境）；驱动 610.88 兼容
- 合并：`llamafactory-cli export configs/merge_lora.yaml` → `/root/autodl-tmp/merged/dpo`（8.3G，2 分片，processor/template 齐全）
- 启动：`sh serving/serve_vllm.sh`（`--served-model-name chartqa-vl --port 8000 --limit-mm-per-prompt '{"image":1}' --mm-processor-kwargs '{"max_pixels":589824}' --max-model-len 4096`；`VLLM_USE_FLASHINFER_SAMPLER=0`）
- 显存占用：**20.7G / 24G**（util 0.85）；启动约 2-3 分钟（含 torch.compile）

## 正确性

| 验证 | 结果 |
| --- | --- |
| 合并等价性（fast800，merged vs adapter） | 87.38% vs 87.25%；23/800 输出文本不同（bf16 舍入），3 题翻变净 +1 |
| 5 题 demo ids 经 vLLM | **5/5 全对**，与云端 pred 一致（Yes / 0.07 / 0.11 / Child (before age 5) / 44） |
| 同批题目准确率（前 6 / 50 / 200 题） | vLLM 50.0 / 64.0 / **78.0%** vs 云端 50.0 / 62.0 / 77.5%（±1 题） |

## 性能（OpenAI `/v1/chat/completions`，temperature=0，max_tokens=64）

| 并发 | 吞吐 | 平均延迟 | p50 | p95 |
| --- | --- | --- | --- | --- |
| 1（200 题） | 9.8 req/s | 0.102s | 0.108s | 0.163s |
| 8（200 题） | **90.4 req/s** | 0.084s | 0.077s | 0.139s |

（对比本机 transformers 8bit 单实例：同题 0.7–6.7s/题——vLLM 的 continuous batching + PagedAttention 把单卡利用率拉开了两个数量级）

## 踩坑

1. vLLM 0.31 的 `--limit-mm-per-prompt` 要 JSON（`'{"image":1}'`），不再接受 `image=1`
2. flashinfer 采样器运行期 JIT 需要 `ninja` 在 PATH；venv 未激活时找不到 → `VLLM_USE_FLASHINFER_SAMPLER=0` 用原生采样器（评测为 greedy，无影响）
3. 镜像源速度实测：阿里 0.14MB/s vs 清华 6.4MB/s（装大依赖前先 `curl -w %{speed_download}` 测速）；vLLM 0.31 依赖很重（torch 2.13 + CUDA13 全栈，下载 ~4GB）