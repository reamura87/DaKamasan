"""Buat dataset config .toml untuk kohya sd-scripts.

Contoh:
    python scripts/make_config.py --image-dir datasets/kamasan --out configs/lora_config.toml
"""
import argparse
import os

TEMPLATE = """[general]
enable_bucket = true
shuffle_caption = true
keep_tokens = {keep_tokens}
caption_dropout_rate = {dropout}

[[datasets]]
resolution = {resolution}
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

  [[datasets.subsets]]
  image_dir = "{image_dir}"
  caption_extension = ".txt"
  num_repeats = {repeats}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--out", default="configs/lora_config.toml")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--keep-tokens", type=int, default=2)
    p.add_argument("--caption-dropout", type=float, default=0.05)
    args = p.parse_args()

    config = TEMPLATE.format(
        keep_tokens=args.keep_tokens,
        dropout=args.caption_dropout,
        resolution=args.resolution,
        image_dir=os.path.abspath(args.image_dir),
        repeats=args.repeats,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(config)
    print(f"Config tersimpan di: {args.out}\n")
    print(config)


if __name__ == "__main__":
    main()
