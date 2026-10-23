"""Auto-caption gambar dataset dengan BLIP, lalu tambahkan trigger words gaya Kamasan.

Caption yang sudah ada dilewati (aman dijalankan ulang saat menambah gambar baru).

Contoh:
    python scripts/caption_dataset.py --image-dir datasets/kamasan
"""
import argparse
import os

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

DEFAULT_TRIGGER = (
    "ptmh style, kamasan balinese painting, traditional balinese art, "
    "flat perspective, intricate details, natural earth tones"
)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", required=True, help="Folder berisi gambar; caption .txt ditulis di folder yang sama")
    p.add_argument("--trigger", default=DEFAULT_TRIGGER, help="Trigger words yang ditambahkan setelah caption BLIP")
    p.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    p.add_argument("--max-new-tokens", type=int, default=50)
    args = p.parse_args()

    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model)

    processed = skipped = 0
    for img_file in sorted(os.listdir(args.image_dir)):
        if not img_file.lower().endswith(IMAGE_EXTS):
            continue
        name = os.path.splitext(img_file)[0]
        caption_path = os.path.join(args.image_dir, f"{name}.txt")
        if os.path.exists(caption_path):
            skipped += 1
            continue

        img = Image.open(os.path.join(args.image_dir, img_file)).convert("RGB")
        inputs = processor(img, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        caption = processor.decode(out[0], skip_special_tokens=True)
        full_caption = f"{caption}, {args.trigger}"

        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(full_caption)
        processed += 1
        print(f"OK {img_file}: {full_caption}")

    print(f"\nSelesai! {processed} caption baru, {skipped} dilewati.")


if __name__ == "__main__":
    main()
