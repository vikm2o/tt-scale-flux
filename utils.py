import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxPipeline
import re
import hashlib
from typing import Dict
import json


# Adapted from Diffusers.
def prepare_latents(
    batch_size: int,
    height: int,
    width: int,
    num_latent_channels: int,
    vae_scale_factor,
    generator: torch.Generator,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    shape = (batch_size, num_latent_channels, height, width)
    latents = randn_tensor(shape, generator=generator, device=torch.device(device), dtype=dtype)
    latents = FluxPipeline._pack_latents(latents, batch_size, num_latent_channels, height, width)
    return latents


def get_noises(
    seed: int,
    max_seed: int,
    num_samples: int,
    height: int,
    width: int,
    num_latent_channels: int,
    vae_scale_factor: int,
    device="cuda",
    dtype=torch.bfloat16,
) -> Dict[int, torch.Tensor]:
    seeds = torch.randint(0, high=max_seed, size=(num_samples,), generator=torch.manual_seed(seed))
    print(f"{seeds=}")
    noises = {
        int(noise_seed): prepare_latents(
            batch_size=1,
            height=height,
            width=width,
            num_latent_channels=num_latent_channels,
            vae_scale_factor=vae_scale_factor,
            generator=torch.manual_seed(int(noise_seed)),
            device=device,
            dtype=dtype,
        )
        for noise_seed in seeds
    }
    assert len(noises) == len(seeds)
    return noises


def load_verifier_prompt(path: str) -> str:
    with open(path, "r") as f:
        verifier_prompt = f.read().replace('"""', "")

    return verifier_prompt


def prompt_to_filename(prompt, max_length=100):
    """ChatGPT, thanks!"""
    # Step 1: Normalize the string by replacing spaces and removing special characters
    filename = re.sub(r"[^a-zA-Z0-9]", "_", prompt.strip())  # Replace non-alphanumeric with '_'
    filename = re.sub(r"_+", "_", filename)  # Collapse multiple underscores
    filename = f"prompt@{filename}"

    if len(filename) > max_length:
        # Use a hash to ensure uniqueness if truncated
        hash_digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        filename = f"{filename[:max_length - 9]}_hash@{hash_digest}"

    return filename


def recover_json_from_output(output: str):
    start = output.find("{")
    end = output.rfind("}") + 1
    json_part = output[start:end]
    return json.loads(json_part)
