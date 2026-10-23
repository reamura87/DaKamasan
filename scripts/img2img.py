"""Perhalus hasil generate lewat img2img dengan LoRA Kamasan.

strength 0.3-0.5 = perubahan halus, 0.6-0.8 = perubahan besar.

Contoh:
    python scripts/img2img.py --lora output/pita_maha_v3_dim64.safetensors \\
        --input output/hasil.png --out output/hasil_img2img.png
"""
import argparse
import os

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image

DEFAULT_PROMPT = (
    "ptmh style, kamasan balinese painting, beautiful symmetrical face, "
    "balinese traditional outfit, almond shaped eyes, defined facial features, "
    "balinese temple garden background, lush tropical trees, warm earth tones, "
    "masterpiece, high detail"
)
DEFAULT_NEGATIVE = (
    "ugly face, deformed face, blurry face, asymmetrical eyes, bad anatomy, "
    "extra limbs, modern, photograph, western art, low quality, blurry"
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lora", required=True)
    p.add_argument("--input", required=True, help="Gambar awal")
    p.add_argument("--out", default="output/hasil_img2img.png")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--strength", type=float, default=0.35)
    p.add_argument("--guidance", type=float, default=9.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    lora_dir, lora_file = os.path.split(os.path.abspath(args.lora))
    pipe.load_lora_weights(lora_dir, weight_name=lora_file)
    pipe = pipe.to(device)

    init_image = Image.open(args.input).convert("RGB").resize((512, 768))
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=init_image,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        clip_skip=2,
        generator=torch.Generator(device).manual_seed(args.seed),
    ).images[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.save(args.out)
    print(f"Tersimpan: {args.out}")


if __name__ == "__main__":
    main()
