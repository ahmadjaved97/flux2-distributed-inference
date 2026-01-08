#!/usr/bin/env python3
"""
Flux2 multi-GPU (sharded) inference via device_map="auto" (Accelerate-style).

Default behavior is quiet and simple.
Use --verbose for extra diagnostics (GPU info, device maps, memory usage).
"""

import argparse
import io
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from huggingface_hub import get_token

from src.flux2_pipeline import Flux2Pipeline


# -----------------------------
# Helpers
# -----------------------------
def _print(msg: str, enabled: bool = True):
    if enabled:
        print(msg, flush=True)


def get_gpu_summary() -> Dict[int, Dict[str, str]]:
    """Lightweight GPU summary (name + total + allocated)."""
    if not torch.cuda.is_available():
        return {}
    info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        alloc_gb = torch.cuda.memory_allocated(i) / (1024**3)
        info[i] = {
            "name": props.name,
            "total_gb": f"{total_gb:.2f}",
            "allocated_gb": f"{alloc_gb:.2f}",
        }
    return info


def print_gpu_summary(prefix: str = "", verbose: bool = False):
    if not verbose:
        return
    g = get_gpu_summary()
    if not g:
        _print(f"{prefix}GPU: not available", True)
        return
    _print(f"{prefix}GPU summary:", True)
    for i, d in g.items():
        _print(f"{prefix}  GPU {i}: {d['name']} | {d['allocated_gb']} / {d['total_gb']} GB", True)


def remote_text_encoder(prompt: str, device: torch.device) -> torch.Tensor:
    """
    Fetch prompt embeddings from a remote API endpoint.
    NOTE: This assumes the endpoint returns a torch-serialized tensor.
    """
    import requests  # only import if used

    url = "https://remote-text-encoder-flux-2.huggingface.co/predict"
    token = get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token not found. Run `huggingface-cli login` or set HF_TOKEN."
        )

    resp = requests.post(
        url,
        json={"prompt": prompt},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    resp.raise_for_status()
    embeds = torch.load(io.BytesIO(resp.content), map_location="cpu")
    return embeds.to(device)


# -----------------------------
# CLI
# -----------------------------
@dataclass
class Args:
    model: str
    prompt: str
    steps: int
    cfg: float
    height: int
    width: int
    seed: int
    out: str
    dtype: str
    remote_encoder: bool
    verbose: bool


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Flux2 sharded (multi-GPU) inference")
    p.add_argument("--model", type=str, default="black-forest-labs/FLUX.2-dev", help="HF repo id or local path")
    p.add_argument("--prompt", type=str, required=False, default="A photorealistic mountain lake at sunset", help="Prompt")
    p.add_argument("--steps", type=int, default=50, help="Denoising steps")
    p.add_argument("--cfg", type=float, default=4.0, help="Guidance scale")
    p.add_argument("--height", type=int, default=1024, help="Image height")
    p.add_argument("--width", type=int, default=1024, help="Image width")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out", type=str, default="out.png", help="Output image path")

    p.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Compute dtype",
    )

    p.add_argument(
        "--remote-encoder",
        action="store_true",
        help="Use remote text encoder (saves local VRAM). Requires HF token.",
    )

    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostics (GPU info, device distribution, memory usage).",
    )

    a = p.parse_args()
    return Args(
        model=a.model,
        prompt=a.prompt,
        steps=a.steps,
        cfg=a.cfg,
        height=a.height,
        width=a.width,
        seed=a.seed,
        out=a.out,
        dtype=a.dtype,
        remote_encoder=a.remote_encoder,
        verbose=a.verbose,
    )


def to_torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script (torch.cuda.is_available() is False).")

    torch_dtype = to_torch_dtype(args.dtype)

    _print("Loading Flux2 pipeline...", True)
    print_gpu_summary(prefix="", verbose=args.verbose)

    # For prompt_embeds mode we need a device to place embeds. Use cuda:0.
    embed_device = torch.device("cuda:0")

    if args.remote_encoder:
        # Don’t load local text encoder
        pipe = Flux2Pipeline.from_pretrained_distributed(
            args.model,
            torch_dtype=torch_dtype,
            text_encoder_device_map=None,
            transformer_device_map="auto",
            vae_device_map=None,
        )

        if args.verbose:
            _print("Using remote text encoder.", True)

        prompt_embeds = remote_text_encoder(args.prompt, device=embed_device)

        if args.verbose:
            dist = pipe.get_device_distribution()
            _print("Device distribution:", True)
            for k, v in dist.items():
                _print(f"  {k}: {'sharded' if isinstance(v, dict) else v}", True)

        _print("Generating...", True)
        image = pipe(
            prompt_embeds=prompt_embeds,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]

    else:
        # Standard: local encoder + sharded device maps
        pipe = Flux2Pipeline.from_pretrained_distributed(
            args.model,
            torch_dtype=torch_dtype,
            text_encoder_device_map="auto",
            transformer_device_map="auto",
            vae_device_map=None,  # keep VAE on a single GPU
        )

        if args.verbose:
            dist = pipe.get_device_distribution()
            _print("Device distribution:", True)
            for k, v in dist.items():
                if isinstance(v, dict):
                    _print(f"  {k}: sharded ({len(v)} layers)", True)
                else:
                    _print(f"  {k}: {v}", True)

        _print("Generating...", True)
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]

    image.save(args.out)
    _print(f"Saved: {args.out}", True)

    if args.verbose:
        print_gpu_summary(prefix="", verbose=True)


if __name__ == "__main__":
    main()