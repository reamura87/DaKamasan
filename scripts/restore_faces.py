"""Perbaiki wajah (dan upscale 2x) dengan GFPGAN.

Contoh:
    python scripts/restore_faces.py --input output/hasil.png --out output/hasil_face_fixed.png
"""
import argparse
import os
import sys
import types
import urllib.request

import cv2
import torchvision.transforms.functional as F

# Patch kompatibilitas: basicsr lama masih meng-import modul yang dihapus
# dari torchvision baru (Python 3.12 + torchvision >= 0.17).
if "torchvision.transforms.functional_tensor" not in sys.modules:
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ImportError:
        shim = types.ModuleType("torchvision.transforms.functional_tensor")
        shim.rgb_to_grayscale = F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = shim

from gfpgan import GFPGANer  # noqa: E402

WEIGHTS_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="output/hasil_face_fixed.png")
    p.add_argument("--weights", default="weights/GFPGANv1.4.pth")
    p.add_argument("--upscale", type=int, default=2)
    p.add_argument("--comparison", help="Opsional: simpan gambar perbandingan sebelum/sesudah")
    args = p.parse_args()

    if not os.path.exists(args.weights):
        os.makedirs(os.path.dirname(os.path.abspath(args.weights)), exist_ok=True)
        print(f"Mengunduh bobot GFPGAN ke {args.weights} ...")
        urllib.request.urlretrieve(WEIGHTS_URL, args.weights)

    restorer = GFPGANer(
        model_path=args.weights, upscale=args.upscale, arch="clean", channel_multiplier=2
    )
    input_img = cv2.imread(args.input)
    if input_img is None:
        raise SystemExit(f"Tidak bisa membaca gambar: {args.input}")

    _, _, restored_img = restorer.enhance(
        input_img, has_aligned=False, only_center_face=False, paste_back=True
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, restored_img)
    print(f"Tersimpan: {args.out}")

    if args.comparison:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Sebelum")
        axes[1].imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Sesudah GFPGAN")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(args.comparison)
        print(f"Perbandingan: {args.comparison}")


if __name__ == "__main__":
    main()
