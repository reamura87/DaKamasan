# Kamasan LoRA — Gaya Lukisan Pita Maha / Kamasan Bali

Fine-tuning **LoRA** pada **Stable Diffusion 1.5** untuk menghasilkan gambar bergaya lukisan tradisional Kamasan (Bali) era Pita Maha: perspektif datar, detail rumit, dan palet warna tanah.

Alur proyek:

1. **Caption otomatis** dataset dengan BLIP + trigger words (`ptmh style, kamasan balinese painting, ...`)
2. **Training LoRA** dengan [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
3. **Generate** gambar (text-to-image) dan **img2img** untuk memperhalus
4. **Perbaikan wajah** + upscale 2x dengan GFPGAN

## Struktur repo

```
kamasan-lora/
├── configs/
│   ├── accelerate_config.yaml   # config accelerate (1 GPU, fp16)
│   └── lora_config.toml         # contoh dataset config kohya
├── datasets/                    # taruh gambar di sini (tidak di-commit)
├── notebooks/
│   └── DaKamasan_experiments.ipynb  # notebook Colab asli (output dibersihkan)
├── output/                      # hasil training & generate (tidak di-commit)
├── scripts/
│   ├── setup.sh                 # clone sd-scripts + install dependensi
│   ├── caption_dataset.py       # auto-caption BLIP
│   ├── check_dataset.py         # cek gambar vs caption
│   ├── make_config.py           # buat dataset config .toml
│   ├── train_lora.sh            # training LoRA
│   ├── generate.py              # text-to-image
│   ├── img2img.py               # img2img
│   └── restore_faces.py         # GFPGAN
├── requirements.txt
└── README.md
```

## Kebutuhan

- Python 3.10+ dan GPU NVIDIA dengan CUDA (dikembangkan di Colab Tesla T4, 16 GB)
- Akun Hugging Face tidak wajib, tetapi model dasar diunduh dari Hugging Face Hub

## Instalasi

```bash
git clone https://github.com/<username>/kamasan-lora.git
cd kamasan-lora

# 1) Pasang PyTorch sesuai CUDA Anda (contoh CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Clone sd-scripts dan install dependensi
bash scripts/setup.sh
```

## Pemakaian

```bash
# 1. Siapkan dataset di datasets/kamasan/ lalu buat caption
python scripts/caption_dataset.py --image-dir datasets/kamasan
python scripts/check_dataset.py   --image-dir datasets/kamasan

# 2. Buat dataset config
python scripts/make_config.py --image-dir datasets/kamasan --out configs/lora_config.toml

# 3. Training
bash scripts/train_lora.sh configs/lora_config.toml output pita_maha_v3_dim64

# 4. Generate
python scripts/generate.py \
  --lora output/pita_maha_v3_dim64.safetensors \
  --prompt "ptmh style, kamasan balinese painting, a balinese dancer performing legong" \
  --out output/hasil.png

# 5. (Opsional) img2img dan perbaikan wajah
python scripts/img2img.py --lora output/pita_maha_v3_dim64.safetensors \
  --input output/hasil.png --out output/hasil_img2img.png
python scripts/restore_faces.py --input output/hasil.png --out output/hasil_face_fixed.png
```

### Hyperparameter training

| Parameter | Nilai |
|---|---|
| Model dasar | `runwayml/stable-diffusion-v1-5` |
| Network | LoRA, dim 64, alpha 32 |
| Learning rate | UNet 5e-5, text encoder 1e-5 |
| Scheduler | cosine_with_restarts (3 siklus), warmup 100 |
| Optimizer | AdamW8bit |
| Epoch | 15 (simpan tiap 3 epoch) |
| Resolusi | 512 dengan bucketing, repeats 12 |
| Lainnya | fp16, gradient checkpointing, clip skip 2, min SNR gamma 5 |

Trigger words: `ptmh style`, `kamasan balinese painting`.

## Menjalankan di Google Colab

Notebook asli ada di `notebooks/DaKamasan_experiments.ipynb`. Notebook ini berisi eksperimen mentah (termasuk percobaan instalasi yang gagal) dan path Google Drive pribadi, sehingga lebih cocok sebagai catatan riwayat. Untuk dijalankan ulang, gunakan script di atas.

## Catatan

- Di Colab dengan PyTorch terbaru, GFPGAN/basicsr butuh patch `torchvision.transforms.functional_tensor`; sudah disertakan di `scripts/restore_faces.py`.
- Dataset dan bobot `.safetensors` tidak disertakan di repo. Pastikan Anda berhak memakai gambar yang dilatih.
- Model dasar Stable Diffusion 1.5 tunduk pada lisensinya sendiri (CreativeML OpenRAIL-M).

## Lisensi

Kode di repo ini berlisensi MIT (lihat `LICENSE`). Lisensi tidak mencakup dataset maupun model pihak ketiga.

## Ucapan terima kasih

[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-large), [TencentARC GFPGAN](https://github.com/TencentARC/GFPGAN).
