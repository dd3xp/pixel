"""Entry point: python scripts/run.py -c configs/petal_16_db32.yaml"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.optimize import run, run_pyramid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-o", "--out", default="workdir")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out_dir = Path(args.out) / Path(args.config).stem / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "config.yaml")

    if "scales" in cfg:
        run_pyramid(cfg, out_dir)
    else:
        run(cfg, out_dir)


if __name__ == "__main__":
    main()
