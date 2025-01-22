import numpy as np
import torch
from diffusers import FluxPipeline
from tqdm.auto import tqdm

from verifier import load_verifier, prepare_inputs, perform_inference
from utils import load_verifier_prompt, prompt_to_filename, get_noises, recover_json_from_output

# Constants
NUM_NOISE_TO_SAMPLE = 8  # Size of initial noise pool
NUM_PROMPTS = 2  # Number of prompts to use for experiments
HEIGHT, WIDTH = 1024, 1024
NUM_LATENT_CHANNELS = 16
SEED = 0
VAE_SCALE_FACTOR = 8
MAX_SEED = np.iinfo(np.int32).max  # To generate random seeds
MAX_NEW_TOKENS = 300  # Maximum number of tokens the verifier can use
TOPK = 3  # Number of maximum noise(s) to start  the search with


def sample(noises, prompt, verifier_prompt, search_round, pipe, verifier, processor, topk):
    images_for_prompt = []
    noises_used = []
    seeds_used = []
    prompt_filename = prompt_to_filename(prompt)
    for i, (seed, noise) in enumerate(noises.items()):
        image = pipe(
            prompt=prompt,
            latents=noise,
            height=HEIGHT,
            width=WIDTH,
            max_sequence_length=512,
            guidance_scale=3.5,
            num_inference_steps=50, # consider decreasing for different search_rounds?
        ).images[0]

        images_for_prompt.append(image)
        noises_used.append(noise)
        seeds_used.append(seed)

        # Save the intermediate image
        filename = f"{prompt_filename}_{search_round}_{seed}.png"
        image.save(filename)

    # Prepare verifier inputs and perform inference
    verifier_inputs = prepare_inputs(
        system_prompt=verifier_prompt,
        images=images_for_prompt,
        prompts=[prompt] * len(images_for_prompt),
        processor=processor,
    )
    outputs = perform_inference(
        model=verifier,
        processor=processor,
        inputs=verifier_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # Convert raw output to JSON and attach noise
    outputs = [recover_json_from_output(o) for o in outputs]
    results = []
    for json_dict, seed_val, noise in zip(outputs, seeds_used, noises_used):
        # Attach the noise tensor so we can select top-K
        merged = {**json_dict, "noise": noise, "seed": seed_val}
        results.append(merged)

    # Sort by 'overall_score' descending and pick top-K
    sorted_list = sorted(results, key=lambda x: x["overall_score"], reverse=True)
    topk_scores = sorted_list[:topk]

    # Update `starting_noises` with the new top-K so next iteration continues the search
    new_noises = {}
    for item in topk_scores:
        top_seed = item["seed"]
        top_noise = item["noise"]
        new_noises[top_seed] = top_noise

    noises = new_noises
    search_round += 1

    # Print some debug info
    for ts in topk_scores:
        print(f"Prompt='{prompt}' | Best seed={ts['seed']} | Score={ts['overall_score']}")
    return noises, search_round

@torch.no_grad()
def main():
    """
    - Samples a pool of random noises.
    - For each text prompt:
      - Generates candidate images with each noise.
      - Passes them through the 'verifier' model to get scores.
      - Saves the top-K noise(s) and updates 'starting_noises' so the search continues.
      - Saves the final, best image for each prompt.
    """
    # --- 1) Sample initial noises
    starting_noises = get_noises(
        seed=SEED,
        max_seed=MAX_SEED,
        height=HEIGHT,
        width=WIDTH,
        num_latent_channels=NUM_LATENT_CHANNELS,
        vae_scale_factor=VAE_SCALE_FACTOR,
        num_samples=NUM_NOISE_TO_SAMPLE,
    )

    # --- 2) Load system prompt and text prompts
    verifier_prompt = load_verifier_prompt("verifier_prompt.txt")
    with open("prompts_open_image_pref_v1.txt", "r") as f:
        prompts = [line.strip() for line in f.readlines()][:NUM_PROMPTS]

    print(f"Using {len(prompts)} prompt(s) and {len(starting_noises)} initial noise(s).")

    # --- 3) Set up the image-generation pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    ).to("cuda:0")

    # --- 4) Load the verifier model and its processor
    verifier, processor = load_verifier()

    # --- 5) Main loop: Generate images, verify, and update noise set
    for prompt in tqdm(prompts, desc="Prompts"):
        noises = starting_noises
        search_round = 0
        topk = TOPK
        while len(noises) >= 1:
            print(f"{len(noises)=}, {topk=}")
            noises, search_round = sample(
                noises=noises, 
                prompt=prompt, 
                verifier_prompt=verifier_prompt, 
                search_round=search_round,
                pipe=pipe, 
                verifier=verifier,
                processor=processor,
                topk=topk,
            )
            topk -= 1

    
if __name__ == "__main__":
    main()
