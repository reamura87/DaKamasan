# Dataset

Letakkan gambar lukisan Kamasan (`.jpg`, `.jpeg`, `.png`) di folder ini, misalnya `datasets/kamasan/`.
Folder ini sengaja tidak di-commit ke Git (lihat `.gitignore`).

Buat caption otomatis:

```bash
python scripts/caption_dataset.py --image-dir datasets/kamasan
python scripts/check_dataset.py   --image-dir datasets/kamasan
```

Pastikan Anda berhak memakai gambar yang Anda latih (hak cipta / lisensi koleksi).
