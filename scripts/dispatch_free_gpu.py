"""
Unified Zero-Cost Cloud GPU Experiment Dispatcher.
Supports Google Colab (1-click notebook), Kaggle Kernels API (2x T4 30h/wk),
Modal Serverless (A10G $30/mo free credit), and Lightning AI Studios.

Usage:
    python scripts/dispatch_free_gpu.py --provider kaggle
    python scripts/dispatch_free_gpu.py --provider colab
    python scripts/dispatch_free_gpu.py --provider modal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.finetuning.azure_finops_dispatcher import get_free_gpu_dispatch_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch Unsloth QLoRA + DPO runs to free GPU services")
    parser.add_argument(
        "--provider",
        choices=["colab", "kaggle", "modal", "lightning", "all"],
        default="all",
    )
    args = parser.parse_args()

    targets = get_free_gpu_dispatch_targets()
    if args.provider != "all":
        targets = [t for t in targets if t.provider_id == args.provider]

    print(json.dumps([t.model_dump() for t in targets], indent=2))


if __name__ == "__main__":
    main()
