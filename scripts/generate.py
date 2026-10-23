"""Generate gambar (text-to-image) dengan Stable Diffusion 1.5 + LoRA Kamasan.

Contoh:
    python scripts/generate.py \\
        --lora output/pita_maha_v3_dim64.safetensors \\
        --prompt "ptmh style, kamasan balinese painting, a balinese dancer" \\
        --out output/hasil.png
"""
import argparse
import os

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

DEFAULT_NEGATIVE = (
    "blurry, low quality, deformed face, ugly face, bad anatomy, extra limbs, "
    "mutated hands, poorly drawn face, disfigured, modern style, photograph, "
    "western art, 3d render, dark background, black background, plain background, "
    "lowres, watermark, signature, jpeg artifacts, cross-eyed, asymmetrical face"
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lora", required=True, help="Path file LoRA .safetensors")
    p.add_argument("--prompt", default="ptmh style, kamasan balinese painting, a balinese dancer")
    p.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--clip-skip", type=int, default=2)
    p.add_argument("--seed", type=int, default=None, help="Kosongkan untuk seed acak")
    p.add_argument("--out", default="output/hasil.png")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    )
    # Scheduler DPM++ Karras: hasil lebih halus dan tajam
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
    )
    lora_dir, lora_file = os.path.split(os.path.abspath(args.lora))
    pipe.load_lora_weights(lora_dir, weight_name=lora_file)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    seed = args.seed if args.seed is not None else torch.randint(0, 999_999, (1,)).item()
    generator = torch.Generator(device).manual_seed(seed)
    print(f"Seed: {seed}")

    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        generator=generator,
        clip_skip=args.clip_skip,
    ).images[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    image.save(args.out)
    print(f"Tersimpan: {args.out}")


if __name__ == "__main__":
    main()
