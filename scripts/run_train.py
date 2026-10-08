import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFT/DPO 训练入口（调用 LLaMA-Factory）")
    ap.add_argument("--config", default=str(ROOT / "configs" / "sft_lora.yaml"))
    ap.add_argument("--output-dir", default=None, help="覆盖 yaml 里的 output_dir")
    ap.add_argument("--dataset", default=None, help="覆盖 yaml 里的 dataset")
    ap.add_argument("--num-epochs", type=float, default=None, help="覆盖 num_train_epochs")
    ap.add_argument("--learning-rate", type=float, default=None, help="覆盖 learning_rate")
    return ap.parse_args()


def build_cmd(args: argparse.Namespace) -> list:
    cmd = ["llamafactory-cli", "train", str(args.config)]
    if args.output_dir:
        cmd += ["--output_dir", args.output_dir]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.num_epochs is not None:
        cmd += ["--num_train_epochs", str(args.num_epochs)]
    if args.learning_rate is not None:
        cmd += ["--learning_rate", str(args.learning_rate)]
    return cmd


def read_output_dir(config_path: Path):
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cfg.get("output_dir")
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        sys.exit(f"找不到配置文件: {config_path}")
    if shutil.which("llamafactory-cli") is None:
        sys.exit("找不到 llamafactory-cli，请确认 conda 环境（base）已加入 PATH")

    cmd = build_cmd(args)
    print("RUN " + " ".join(cmd), flush=True)
    t0 = time.time()
    ret = subprocess.run(cmd).returncode
    minutes = (time.time() - t0) / 60

    if ret != 0:
        sys.exit(f"训练失败（退出码 {ret}），详见上方日志")
    output_dir = args.output_dir or read_output_dir(config_path)
    print(f"DONE 训练结束，用时 {minutes:.1f} 分钟", flush=True)
    if output_dir:
        print(f"adapter: {output_dir}", flush=True)
    print("下一步: sh test.sh", flush=True)


if __name__ == "__main__":
    main()