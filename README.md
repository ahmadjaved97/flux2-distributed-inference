# Flux2 Distributed Inference

**Unofficial multi-GPU (sharded/model-parallel) FLUX.2 inference using 🤗 Accelerate `device_map="auto"`.**  
Runs FLUX.2 locally by automatically sharding large components across multiple GPUs. Optional remote text encoder supported.

---

## 🎯 Motivation

Running **FLUX.2** locally is often GPU-memory bound. Common workarounds include:

1. **Quantization** — can degrade image quality
2. **Remote text encoders** — requires internet + adds latency
3. **CPU offloading** — works but is slow

**This repo enables multi-GPU sharded inference via `device_map="auto"`**, meaning:

- ✅ **No quantization** — full bf16/fp16 quality
- ✅ **Local by default** — remote text encoder is optional
- ✅ **Multi-GPU friendly** — distributes model weights across available GPUs
- ✅ **Simple API** — drop-in `from_pretrained_distributed(...)`

> Note: This is **model-parallel sharding** (weights split across GPUs), not data-parallel batching (DDP).

---

## 🚀 Installation

```bash
# Clone the repository
git clone git@github.com:YigitEkin/flux2-distributed-inference.git
cd flux2-distributed-inference

# Install dependencies
conda create -n flux2dist python=3.10 -y
conda activate flux2dist
pip install -r requirements.txt

# Login to Hugging Face (required for FLUX.2 model access)
huggingface-cli login
```

---

## ⚡ Quick Start

**Minimal Example**

```python
import torch
from src.flux2_pipeline import Flux2Pipeline

pipe = Flux2Pipeline.from_pretrained_distributed(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
    text_encoder_device_map="auto",      # shard text encoder
    transformer_device_map="auto",       # shard transformer
    vae_device_map=None,                 # keep VAE on one GPU (default)
)

image = pipe(
    prompt="A hyperreal macro photo of a glass marble on wet asphalt, neon reflections, shallow depth of field",
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

image.save("output.png")
print("Saved output.png")
```

**CLI Usage**

```bash
# Basic usage
python inference.py --prompt "A photorealistic mountain lake at sunset"

# Customize generation
python inference.py \
    --prompt "A cyberpunk cityscape at night, neon lights, rain" \
    --steps 50 \
    --cfg 4.0 \
    --height 1024 \
    --width 1024 \
    --seed 42 \
    --out result.png \
    --verbose

# Use remote text encoder (saves VRAM, requires HF token)
python inference.py \
    --prompt "Your prompt here" \
    --remote-encoder \
    --verbose
```
---

## 🔍 How It Works

The `from_pretrained_distributed` class method leverages 🤗 **Accelerate’s** `device_map="auto"` to automatically shard large model components across multiple GPUs:

```python
@classmethod
def from_pretrained_distributed(
    cls,
    pretrained_model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    max_memory: Optional[Dict[int, str]] = None,
    text_encoder_device_map: Optional[Union[str, Dict]] = "auto",
    transformer_device_map: Optional[Union[str, Dict]] = "auto",
    vae_device_map: Optional[Union[str, Dict]] = None,
    **kwargs,
)
"""
- text_encoder_device_map="auto" shards the text encoder across GPUs
- transformer_device_map="auto" shards the transformer across GPUs
- vae_device_map=None keeps the VAE on one GPU (usually fine)
"""
```
---

## ⚙️ Configuration Options

**Device Map Options**

| Option	| Description |
| --- | --- |
| "auto" | Automatically shard across available GPUs |
| None | Don’t load this component (useful with remote encoder) |
| Dict | Custom device mapping |
| max_memory | Per-GPU memory budget (can allow CPU spill if you add "cpu": "XXGB") |

**Custom Memory Allocation**

```python
pipe = Flux2Pipeline.from_pretrained_distributed(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
    max_memory={0: "35GB", 1: "35GB", 2: "35GB", 3: "35GB", "cpu": "50GB"},
    text_encoder_device_map="auto",
    transformer_device_map="auto",
)
```

---

## 🌐 Using Remote Text Encoder (Optional)

If you want to save VRAM by offloading text encoding to Hugging Face’s remote endpoint:

```python
import io
import torch
import requests
from huggingface_hub import get_token
from src.flux2_pipeline import Flux2Pipeline

PROMPT = "Your prompt here"

pipe = Flux2Pipeline.from_pretrained_distributed(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
    text_encoder_device_map=None,     # don't load local text encoder
    transformer_device_map="auto",
)

response = requests.post(
    "https://remote-text-encoder-flux-2.huggingface.co/predict",
    json={"prompt": PROMPT},
    headers={"Authorization": f"Bearer {get_token()}", "Content-Type": "application/json"},
)
prompt_embeds = torch.load(io.BytesIO(response.content), map_location="cpu").to("cuda:0")

image = pipe(
    prompt_embeds=prompt_embeds,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=4.0,
).images[0]

image.save("remote_output.png")
```

---

## 📊 Measured Peak VRAM (bf16)

These numbers are **measured peaks** from this repo on:

- **Model:** `black-forest-labs/FLUX.2-dev`
- **Precision:** bf16
- **Resolution:** 1024×1024
- **Batch:** 1
- **Text encoder:** local
- **Placement:** `device_map="auto"`
- **Hardware tested:** **NVIDIA H100 80GB HBM3**
- **Driver/CUDA:** **Driver 570.148.08**, **CUDA 12.8**

> Peak VRAM can vary with driver/CUDA, PyTorch, diffusers version, attention backend, and whether any layers spill to CPU.

### Peak VRAM by #GPUs (H100 80GB)

| GPUs | Peak VRAM per GPU (GB) | Notes |
|------|-------------------------|------|
| **2 GPUs** | **GPU0 54.96**, **GPU1 53.52** | Works on 2×80GB comfortably |
| **3 GPUs** | **GPU0 37.66**, **GPU1 38.61**, **GPU2 33.36** | More headroom + better balance |
| **4 GPUs** | **GPU0 28.95**, **GPU1 30.02**, **GPU2 30.02**, **GPU3 21.79** | Best headroom / lowest peak |

To reproduce these numbers, run with `--verbose` and look for **“Max GPU Memory Usage (peak)”**.

---
## 🙏 Acknowledgments
This repository is based on the following repositories:

- Black Forest Labs for FLUX.2
- 🤗 Hugging Face for Diffusers, Transformers, and Accelerate
- Mistral AI for the text encoder architecture

---

## 📄 License

This project is licensed under the Apache License 2.0.
See the FLUX.2 model card for FLUX.2 licensing terms.
