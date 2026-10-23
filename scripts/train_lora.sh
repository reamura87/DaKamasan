#!/usr/bin/env bash
# Training LoRA Stable Diffusion 1.5 dengan kohya-ss/sd-scripts.
#
# Pemakaian:
#   bash scripts/train_lora.sh [DATASET_CONFIG] [OUTPUT_DIR] [OUTPUT_NAME]
#
# Variabel lingkungan opsional:
#   SD_SCRIPTS_DIR  lokasi clone sd-scripts (default: ./sd-scripts)
#   BASE_MODEL      model dasar (default: runwayml/stable-diffusion-v1-5)
set -euo pipefail

DATASET_CONFIG="${1:-configs/lora_config.toml}"
OUTPUT_DIR="${2:-output}"
OUTPUT_NAME="${3:-pita_maha_v3_dim64}"
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-sd-scripts}"
BASE_MODEL="${BASE_MODEL:-runwayml/stable-diffusion-v1-5}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_config.yaml}"

if [ ! -d "$SD_SCRIPTS_DIR" ]; then
  echo "sd-scripts tidak ditemukan di '$SD_SCRIPTS_DIR'. Jalankan: bash scripts/setup.sh" >&2
  exit 1
fi

# Path dijadikan absolut karena kita pindah ke folder sd-scripts
DATASET_CONFIG="$(realpath "$DATASET_CONFIG")"
ACCELERATE_CONFIG="$(realpath "$ACCELERATE_CONFIG")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

cd "$SD_SCRIPTS_DIR"

accelerate launch --config_file "$ACCELERATE_CONFIG" \
  train_network.py \
  --pretrained_model_name_or_path="$BASE_MODEL" \
  --dataset_config="$DATASET_CONFIG" \
  --output_dir="$OUTPUT_DIR" \
  --output_name="$OUTPUT_NAME" \
  \
  --network_module=networks.lora \
  --network_dim=64 \
  --network_alpha=32 \
  \
  --learning_rate=5e-5 \
  --unet_lr=5e-5 \
  --text_encoder_lr=1e-5 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=100 \
  --lr_scheduler_num_cycles=3 \
  \
  --optimizer_type="AdamW8bit" \
  --train_batch_size=1 \
  --max_train_epochs=15 \
  --save_every_n_epochs=3 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --save_model_as=safetensors \
  --clip_skip=2 \
  --min_snr_gamma=5 \
  --seed=42
