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