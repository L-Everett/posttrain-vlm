# Technical Notes

Design notes and FAQ for `posttrain-vlm`. Each entry captures a decision, why it was made, and the underlying mechanism.

## 1. Why LoRA instead of full fine-tuning?

- Full fine-tuning a 4B VLM needs well over 80 GB of GPU memory (weights + gradients + optimizer states in bf16); LoRA trains less than 1–2% of parameters and fits a 24 GB card.
- For domain adaptation on thousands of samples, LoRA quality is close to full fine-tuning while producing a small adapter artifact (tens to hundreds of MB) that is trivial to version and merge.
- Fewer trainable parameters also reduce overfitting risk on a narrow domain.

## 2. What do rank and alpha control?

- `r` (rank) is the dimension of the low-rank update `ΔW = B·A`. Larger `r` means more capacity and more adapter parameters.
- `alpha` scales the update: the effective scaling is `alpha / r` when merging. Common practice is `alpha = 2r`.
- For domain adaptation with thousands of samples, `r = 8–32` is usually sufficient; `r = 16` is the default here.

## 3. What does a multimodal SFT sample look like?

- ShareGPT-style records: a `messages` list of `{role, content}` turns plus an `images` field with image paths.
- The content contains an `<image>` placeholder where the vision tokens are inserted by the chat template.
- Dataset registration maps a dataset name to its file and format so the training config can reference it.

## 4. Why must the chat template match between training and inference?

- Special tokens delimit turns and image placeholders. If training and inference use different templates, token positions shift and the model sees a distribution it was not trained on.
- Symptom of a mismatch: fluent but irrelevant answers, or the model echoing the prompt format.

## 5. Why relaxed accuracy for ChartQA?

- Chart answers may legitimately be written as `17.5`, `17.50`, or `17.5%`. Exact string match unfairly penalizes these.
- Relaxed accuracy counts a prediction as correct when it is within 5% relative tolerance of the reference value (the standard ChartQA metric).
- Implementation detail: parse the first number from the generated text, then compare numerically.

## 6. Why build preference pairs by rejection sampling?

- The task has ground-truth answers, so the SFT model can be sampled on training prompts; wrong generations become the `rejected` side and the ground truth is the `chosen` side.
- This produces domain-specific preference pairs without human annotation and is far cheaper than general-purpose preference datasets.
- It also directly targets the model's actual failure modes instead of generic style preferences.

## 7. How does DPO work, and how is it different from PPO?

- DPO optimizes a closed-form loss directly on preference pairs. The implicit reward compares the policy's log-probability ratio against a frozen reference model for chosen vs rejected responses.
- `beta` controls how far the policy may drift from the reference; a typical value here is `0.1`.
- PPO requires training a separate reward model and running an online RL loop — more moving parts and harder to stabilize. With static, verifiable preference pairs, DPO is the simpler and more reliable choice.

## 8. What is the "data flywheel" in this project?

- Evaluate → categorize errors → add targeted training data → retrain → re-evaluate.
- The point is that evaluation output is not just a number; it becomes the next batch of training data.
- Round 2 SFT uses data augmented around the dominant error categories found in round 1.

## 9. Why merge LoRA before serving?

- Merging computes `W' = W + B·A` exactly, removing adapter dispatch overhead and producing a standard checkpoint.
- The merged model works with any inference stack (vLLM, TensorRT-LLM, llama.cpp) without adapter support.
- The adapter remains in the repo for reproducibility; the merged checkpoint is a build artifact.

## 10. Why vLLM for serving and evaluation?

- PagedAttention manages the KV cache in blocks, and continuous batching keeps the GPU busy across requests — both matter for batch evaluation and for the demo.
- Serving through an OpenAI-compatible endpoint keeps the Gradio demo decoupled from the model process.

## 11. Catastrophic forgetting — what to watch?

- A VLM fine-tuned on a narrow domain can lose general abilities (OCR, scene understanding, instruction following).
- Mitigations: keep epochs low, use a modest LoRA rank, and spot-check general capabilities before and after training.
- If degradation appears, mix a small amount of general instruction data into the SFT set.

## 12. Why cap image resolution / visual tokens?

- Visual tokens dominate sequence length: a high-resolution image can contribute thousands of tokens, and activation memory scales with sequence length.
- Capping the visual token budget (roughly 256–1280 per image) keeps larger batch sizes feasible on 24 GB with a small accuracy cost.
- The cap is controlled by pixel limits in the image processor and must be consistent between training and evaluation.