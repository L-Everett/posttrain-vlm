import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = "Please answer with a single value."
DEFAULT_MAX_PIXELS = 768 * 768
DEFAULT_BASE = os.environ.get("MODEL_PATH", "path/to/your/Qwen3-VL-4B-Instruct")
DEFAULT_ADAPTER = os.environ.get("ADAPTER_PATH", "path/to/your/dpo_adapter")
SAMPLES_DIR = ROOT / "serving" / "demo_samples"
FALLBACK_SMOKE = ROOT / "data" / "smoke"


def parse_args():
    ap = argparse.ArgumentParser(description="本机 4bit 对比 demo：原始 Qwen3-VL-4B vs 微调版（关/开 LoRA adapter）")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-4bit", action="store_true", help="不做量化（bf16，需 ~8.3G 显存，仅供调试）")
    ap.add_argument("--load-4bit", action="store_true", help="用 NF4 4bit（更省显存；精度略低于默认 8bit）")
    ap.add_argument("--smoke", action="store_true", help="命令行自检对照，不开网页")
    return ap.parse_args()


def fit_image(image, max_pixels):
    if max_pixels and image.width * image.height > max_pixels:
        factor = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * factor), int(image.height * factor)))
    return image


class PairModel:
    def __init__(self, args):
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.base_model)
        self.processor.tokenizer.padding_side = "left"
        template_path = Path(args.adapter) / "chat_template.jinja"
        if template_path.exists():
            self.processor.chat_template = template_path.read_text(encoding="utf-8")
            print(f"chat template: {template_path}")
        kwargs = {}
        if args.no_4bit:
            print("未量化加载（bf16，显存需求 ~8.3G）")
        elif args.load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForImageTextToText.from_pretrained(args.base_model, dtype="auto", device_map="cuda:0", **kwargs)
        adapter_path = Path(args.adapter)
        if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(model, str(adapter_path))
            self.has_adapter = True
            print(f"adapter: {adapter_path}")
        else:
            self.model = model
            self.has_adapter = False
            print(f"未找到 adapter（{adapter_path}），两侧输出将相同")
        self.model.eval()

    def ask(self, image, question, use_adapter, use_instruction=True):
        prompt = f"{question} {INSTRUCTION}" if use_instruction else question
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=[fit_image(image, self.args.max_pixels)], text=[text], return_tensors="pt", padding=True).to(self.model.device, torch.bfloat16)
        t0 = time.time()
        with torch.no_grad():
            if use_adapter or not self.has_adapter:
                gen = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False)
            else:
                with self.model.disable_adapter():
                    gen = self.model.generate(**inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
        return reply, time.time() - t0


def load_smoke_cases():
    samples_file = SAMPLES_DIR / "samples.json"
    if samples_file.exists():
        cases = json.loads(samples_file.read_text(encoding="utf-8"))
        for c in cases:
            c["_dir"] = SAMPLES_DIR
        return cases
    cases_file = FALLBACK_SMOKE / "infer_cases.json"
    if cases_file.exists():
        cases = json.loads(cases_file.read_text(encoding="utf-8"))
        for c in cases:
            c["_dir"] = FALLBACK_SMOKE
        return cases
    return []


def run_smoke(model):
    cases = load_smoke_cases()
    if not cases:
        print("没有可用的自检样例（serving/demo_samples/samples.json 或 data/smoke/infer_cases.json）")
        return
    print(f"自检样例：{len(cases)} 条（来源 {'demo_samples' if (SAMPLES_DIR / 'samples.json').exists() else 'data/smoke'}）")
    for i, case in enumerate(cases, 1):
        image = Image.open(case["_dir"] / case["image"]).convert("RGB")
        base, tb = model.ask(image, case["question"], use_adapter=False)
        ft, tf = model.ask(image, case["question"], use_adapter=True)
        line = f"[{i}] Q: {case['question'][:80]} | gold={case.get('answer', case.get('gold', '?'))} | 原始={base} ({tb:.1f}s) | 微调={ft} ({tf:.1f}s)"
        if "base_pred" in case or "ft_pred" in case:
            line += f" | 云端对照: 原始={case.get('base_pred', '?')} 微调={case.get('ft_pred', '?')}"
        print(line, flush=True)


def build_ui(model):
    import gradio as gr

    def run(image, question, free_mode):
        if image is None or not question.strip():
            return "", "", "请先提供图片和问题"
        base, tb = model.ask(image, question, use_adapter=False, use_instruction=not free_mode)
        ft, tf = model.ask(image, question, use_adapter=True, use_instruction=not free_mode)
        mode = "自由提问" if free_mode else "评测口径（单值指令）"
        return base, ft, f"{mode} ｜ 原始 {tb:.1f}s ｜ 微调 {tf:.1f}s"

    examples = []
    for case in load_smoke_cases()[:8]:
        image_path = str(case["_dir"] / case["image"])
        examples.append([image_path, case["question"]])

    with gr.Blocks(title="ChartQA 微调对比 demo") as demo:
        gr.Markdown(
            "## ChartQA 微调对比：原始 Qwen3-VL-4B vs SFT r2 + DPO\n"
            "上传图表图片、输入问题，两侧同时生成，直观对比微调效果。（本地量化推理，默认 8bit，接近云端 bf16 口径）\n\n"
            "> 提示：这是 ChartQA **单值问答**模型——解释/闲聊类问题会被压成一个值；想看它如何应对开放问题，请勾选「自由提问模式」。"
            "切换例题会同时替换图片与问题；编辑后请点「对比生成」（回车不触发）。"
        )
        with gr.Row():
            image_in = gr.Image(type="pil", label="图表图片", height=420)
            question_in = gr.Textbox(label="问题", value="What is the value of Q2?", lines=3)
        free_mode = gr.Checkbox(label="自由提问模式（不追加单值指令，观察模型对开放问题的反应）", value=False)
        btn = gr.Button("对比生成", variant="primary")
        with gr.Row():
            out_base = gr.Textbox(label="原始 Qwen3-VL-4B（未微调）", lines=3)
            out_ft = gr.Textbox(label="微调后（SFT r2 + DPO）", lines=3)
        status = gr.Textbox(label="状态", interactive=False)
        btn.click(run, [image_in, question_in, free_mode], [out_base, out_ft, status])
        if examples:
            gr.Examples(examples=examples, inputs=[image_in, question_in])
    return demo


def main():
    args = parse_args()
    model = PairModel(args)
    if args.smoke:
        run_smoke(model)
        return
    demo = build_ui(model)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()