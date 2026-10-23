#!/usr/bin/env bash
# Setup lingkungan: clone kohya sd-scripts dan install dependensi.
# Dijalankan dari root repo:  bash scripts/setup.sh
set -euo pipefail

SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-sd-scripts}"

if [ ! -d "$SD_SCRIPTS_DIR" ]; then
  git clone https://github.com/kohya-ss/sd-scripts "$SD_SCRIPTS_DIR"
fi

# Pasang PyTorch sesuai CUDA Anda lebih dulu, mis. untuk CUDA 12.1:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r "$SD_SCRIPTS_DIR/requirements.txt"

echo "Selesai. Lanjut: caption dataset -> make_config -> train_lora."
