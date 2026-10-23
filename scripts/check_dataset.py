"""Cek bahwa setiap gambar punya file caption .txt.

Contoh:
    python scripts/check_dataset.py --image-dir datasets/kamasan
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", required=True)
    args = p.parse_args()

    files = os.listdir(args.image_dir)
    images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    captions = {f for f in files if f.endswith(".txt")}

    print(f"Jumlah gambar  : {len(images)}")
    print(f"Jumlah caption : {len(captions)}")
    print(f"Path dataset   : {args.image_dir}")

    missing = [i for i in images if f"{os.path.splitext(i)[0]}.txt" not in captions]
    if missing:
        print(f"\n{len(missing)} gambar TANPA caption:")
        for m in missing[:10]:
            print(f"   - {m}")
        sys.exit(1)
    print("\nSemua gambar sudah punya caption.")


if __name__ == "__main__":
    main()
