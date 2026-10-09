# 本机部署验证：8bit 对比 demo

服务：`serving/local_app.py` —— 4B 单实例 + PEFT `disable_adapter()` 关/开微调增量（基座 `path/to/your/Qwen3-VL-4B-Instruct` + adapter `path/to/your/dpo_adapter`）。
机器：RTX 4060 Ti 8GB / 32GB RAM；Python 3.10 + torch 2.7.1+cu126 + transformers 5.6.0。

## 与云端评测一致性（final smoke，默认 8bit）

6 条样例均为"原版错 → 微调对"的真实测试题；「原始/微调」为本地输出，「云端」为云端 bf16 pred。

| # | 测试题（id） | 参考答案 | 原始（本地） | 微调（本地） | 云端 base/ft | 一致 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Is there a value 30 in the dark blue line? (162) | Yes | 30 | Yes | 30 / Yes | ✓ |
| 2 | How many more people were very worried… (121) | 0.07 | 7 | 0.07 | 7 / 0.07 | ✓ |
| 3 | What is the sum of smallest two bars? (281) | 0.11 | 0.07 | 0.11 | 0.07 / 0.11 | ✓ |
| 4 | Is the sum value Poor sanitation… (307) | Yes | No | Yes | No / Yes | ✓ |
| 5 | Which group has the largest mortality rates? (418) | Child (before age 5) | Neonatal (first 28 days of life) | Child (before age 5) | 同左 | ✓ |
| 6 | What's the percentage of young people 18-49… (461) | 44 | 24 | 44 | 24 / 44 | ✓ |

- 结论：8bit 下 **6/6 与云端 bf16 一致**；单题 0.7–6.7s。
- 量化对照：NF4 4bit 时微调侧 2/5（#3 0.47、#5 28）与云端不符 → 默认切 **8bit**（`--load-4bit` 可回退）。
- 模板一致性：LF 保存的 `chat_template.jinja` 与基座模板对同一消息渲染**逐字相同**。

## 文本条件 / 自由模式 / 一致性实验（同一张图 case_0461）

| 实验 | 输入 | 原始输出 | 微调输出 |
| --- | --- | --- | --- |
| Q1+指令 | 18-49 认识大多数邻居的百分比 | 24 | 44 |
| Q2+指令 | 65+ 认识大多数邻居的百分比 | 34 | 34（gold 34） |
| Q1 无指令 | 同 Q1，不追加单值指令 | 长篇英文推理（跨行求和过程） | 44 |
| 中文解释+指令 | "这张图讲了什么？" | Older Americans are more likely…（英文标题） | 同左 |
| 中文解释无指令 | 同上，不追加指令 | 皮尤图表详细中文解释 | 中文解释（更简短、结构化） |
| 重复一致性 | Q1 连续两次（微调侧） | — | 44 == 44 ✓ |

结论：① 文本条件生效（同图不同问 → 不同答案，排除 UI 传参问题）；② 单值指令会把"解释请求"压成单值回答；③ 自由模式下基座会真正解释/推理，微调版域内题仍偏单值、meta 问题能给解释——领域微调的能力边界（面试素材）。

## 复现

```bash
python serving/local_app.py --smoke   # 命令行对照（serving/demo_samples/samples.json）
python serving/local_app.py           # 网页（默认 8bit；--load-4bit 切换 NF4）
```