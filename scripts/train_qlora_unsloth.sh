#!/usr/bin/env bash
# Production Linux Shell Script: Unsloth 4-Bit QLoRA Fine-Tuning & Evaluation
set -euo pipefail

LORA_R="${1:-16}"
LORA_ALPHA="${2:-32}"
MAX_SEQ_LEN="${3:-32768}"

echo "[PolicyMiner] Starting Unsloth 4-bit QLoRA pipeline (r=${LORA_R}, alpha=${LORA_ALPHA}, seq=${MAX_SEQ_LEN})..."
python -c "
from src.core.schemas import UnslothQLoRAConfig
from src.finetuning.unsloth_trainer import run_unsloth_qlora_experiment
summary = run_unsloth_qlora_experiment(UnslothQLoRAConfig(lora_r=${LORA_R}, lora_alpha=${LORA_ALPHA}, max_seq_length=${MAX_SEQ_LEN}))
print(summary.model_dump_json(indent=2))
"
echo "[PolicyMiner] Executing 4-Gate MLOps Evaluation Suite..."
python -m src.evals.runner
