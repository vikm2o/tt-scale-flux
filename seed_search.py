import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Union, Optional, Any

import numpy as np
import torch
from diffusers import DiffusionPipeline
from tqdm.auto import tqdm

from utils import (
    generate_neighbors,
    prompt_to_filename,
    get_noises,
    TORCH_DTYPE_MAP,
    get_latent_prep_fn,
    serialize_artifacts,
    MODEL_NAME_MAP,
)
from verifiers import SUPPORTED_VERIFIERS

# Non-configurable constants
TOPK = 1
MAX_SEED = np.iinfo(np.int32).max


def sample(
    noises: dict[int, torch.Tensor],
    prompt: str,
    search_round: int,
    pipe: DiffusionPipeline,
    verifier,
    topk: int,
    root_dir: str,
    config: dict,
) -> dict:
    """
    Generate images from noises, score them with verifier, and return best results.
    """
    use_low_gpu_vram = config.get("use_low_gpu_vram", False)
    batch_size_for_img_gen = config.get("batch_size_for_img_gen", 1)
    verifier_args = config.get("verifier_args")
    choice_of_metric = verifier_args.get("choice_of_metric", "overall_score")
    verifier_to_use = verifier_args.get("name", "gemini")
    search_args = config.get("search_args", None)

    images_for_prompt = []
    noises_used = []
    seeds_used = []
    images_info = []
    prompt_filename = prompt_to_filename(prompt)

    noise_items = list(noises.items())

    for i in range(0, len(noise_items), batch_size_for_img_gen):
        batch = noise_items[i: i + batch_size_for_img_gen]
        seeds_batch, noises_batch = zip(*batch)
        filenames_batch = [
            os.path.join(root_dir, f"{prompt_filename}_r{search_round}_s{seed}.png") for seed in seeds_batch
        ]

        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cuda:0")
        print(f"Generating images for batch with seeds: {list(seeds_batch)}")

        batched_prompts = [prompt] * len(noises_batch)
        batched_latents = torch.stack(noises_batch).squeeze(dim=1)

        batch_result = pipe(prompt=batched_prompts, latents=batched_latents, **config["pipeline_call_args"])
        batch_images = batch_result.images
        if use_low_gpu_vram and verifier_to_use != "gemini":
            pipe = pipe.to("cpu")

        for seed, noise, image, filename in zip(seeds_batch, noises_batch, batch_images, filenames_batch):
            image.save(filename)
            images_for_prompt.append(image)
            noises_used.append(noise)
            seeds_used.append(seed)
            images_info.append((seed, noise, image, filename))

    verifier_inputs = verifier.prepare_inputs(
        images=images_for_prompt, prompts=[prompt] * len(images_for_prompt)
    )
    print("Scoring with the verifier")
    outputs = verifier.score(inputs=verifier_inputs)

    results = []
    for json_dict, seed_val, noise in zip(outputs, seeds_used, noises_used):
        merged = {**json_dict, "noise": noise, "seed": seed_val}
        results.append(merged)

    def get_score(x):
        if isinstance(x[choice_of_metric], dict):
            return x[choice_of_metric]["score"]
        return x[choice_of_metric]

    sorted_list = sorted(results, key=lambda x: get_score(x), reverse=True)
    topk_scores = sorted_list[:topk]

    for ts in topk_scores:
        score_value = get_score(ts)
        print(f"Prompt='{prompt}' | Round={search_round} | Seed={ts['seed']} | Score={score_value:.4f}")

    best_img_path = os.path.join(root_dir, f"{prompt_filename}_r{search_round}_s{topk_scores[0]['seed']}.png")
    datapoint = {
        "prompt": prompt,
        "search_round": search_round,
        "num_noises": len(noises),
        "best_noise_seed": topk_scores[0]['seed'],
        "best_noise": topk_scores[0]["noise"],
        "best_score": get_score(topk_scores[0]),
        "choice_of_metric": choice_of_metric,
        "best_img_path": best_img_path,
        "all_results": sorted_list,
    }

    search_method = search_args.get("search_method", "random") if search_args else "random"
    if search_method == "zero-order":
        first_score = get_score(results[0])
        neighbors_with_better_score = any(get_score(item) > first_score for item in results[1:])
        datapoint["neighbors_improvement"] = neighbors_with_better_score

    # Serialize artifacts if configured
    if config.get("serialize_results", True):
        if search_method == "zero-order":
            if datapoint.get("neighbors_improvement", False):
                serialize_artifacts(images_info, prompt, search_round, root_dir, datapoint)
            else:
                print("Skipping serialization as there was no improvement in this round.")
        else:
            serialize_artifacts(images_info, prompt, search_round, root_dir, datapoint)

    return datapoint


def run_seed_search(
    prompt: str,
    config: Dict[str, Any] = None,
    output_dir: Optional[str] = None,
    pipeline: Optional[DiffusionPipeline] = None,
    verifier = None,
    return_pipe: bool = False,
) -> Dict[str, Any]:
    """
    Run seed search for a given prompt using the specified configuration.
    
    Args:
        prompt: The prompt to use for image generation
        config: Configuration dictionary (optional)
        output_dir: Directory for output files (optional)
        pipeline: Pre-initialized diffusion pipeline (optional)
        verifier: Pre-initialized verifier (optional)
        return_pipe: Whether to return the pipeline object (for reuse)
    
    Returns:
        Dictionary containing best seed results
    """
    # Default config if not provided
    if config is None:
        config = {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "torch_dtype": "float16",
            "use_low_gpu_vram": False,
            "batch_size_for_img_gen": 4,
            "search_args": {
                "search_method": "zero-order",
                "search_rounds": 3,
                "threshold": 0.01,
                "num_neighbors": 8,
            },
            "verifier_args": {
                "name": "gemini",
                "choice_of_metric": "overall_score",
            },
            "pipeline_call_args": {
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            },
        }

    # Create output directory
    if output_dir is None:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.get("pretrained_model_name_or_path", "sd-xl")
        model_key = MODEL_NAME_MAP.get(model_name, model_name.split("/")[-1])
        verifier_name = config["verifier_args"]["name"]
        choice_of_metric = config["verifier_args"]["choice_of_metric"]
        
        output_dir = os.path.join(
            "seed_search",
            model_key,
            verifier_name,
            choice_of_metric,
            prompt_to_filename(prompt)[:20],
            current_datetime,
        )
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # Save the config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        # Create a copy to avoid modifying the original
        save_config = {**config}
        # Remove non-serializable objects
        if "pipe" in save_config:
            del save_config["pipe"]
        if "verifier" in save_config:
            del save_config["verifier"]
        
        json.dump(save_config, f, indent=4)

    # Extract search parameters
    search_args = config["search_args"]
    search_rounds = search_args["search_rounds"]
    search_method = search_args.get("search_method", "random")
    
    # Setup pipeline if not provided
    pipe_provided = pipeline is not None
    if not pipe_provided:
        torch_dtype = TORCH_DTYPE_MAP.get(config.get("torch_dtype", "float16"), torch.float16)
        model_name = config.get("pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0")
        
        pipeline = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        
        lora_file = config.get("lora_file")
        if lora_file and os.path.exists(lora_file):
            print(f"Loading LoRA weights from {lora_file}")
            pipeline.load_lora_weights(lora_file, adapter_name="lora")
        
        if not config.get("use_low_gpu_vram", False):
            pipeline = pipeline.to("cuda:0")
            
        pipeline.set_progress_bar_config(disable=True)
    
    # Setup verifier if not provided
    verifier_provided = verifier is not None
    if not verifier_provided:
        verifier_args = config["verifier_args"]
        verifier_cls = SUPPORTED_VERIFIERS.get(verifier_args["name"])
        if verifier_cls is None:
            raise ValueError(f"Verifier {verifier_args['name']} not available. Check dependencies.")
        
        verifier = verifier_cls(**verifier_args)

    # Track best results across all rounds
    all_rounds_results = []
    best_datapoint = None
    best_datapoint_per_round = {}
    
    # Main search loop
    search_round = 1
    while search_round <= search_rounds:
        # Determine number of noise samples
        if search_method == "zero-order":
            num_noises_to_sample = 1
        else:
            num_noises_to_sample = 2**search_round
        
        print(f"\n=== Prompt: {prompt} | Round: {search_round}/{search_rounds} ===")
        
        # Generate noise pool
        should_regenerate_noise = True
        previous_round = search_round - 1
        
        if previous_round in best_datapoint_per_round:
            was_improvement = best_datapoint_per_round[previous_round].get("neighbors_improvement", False)
            if was_improvement:
                should_regenerate_noise = False
        
        # Generate or reuse noise based on improvement in previous round
        if should_regenerate_noise:
            if search_method == "zero-order" and search_round != 1:
                print("Regenerating base noise because the previous round had no improvement")
            noises = get_noises(
                max_seed=MAX_SEED,
                num_samples=num_noises_to_sample,
                height=config["pipeline_call_args"]["height"],
                width=config["pipeline_call_args"]["width"],
                dtype=torch_dtype if 'torch_dtype' in locals() else torch.float16,
                fn=get_latent_prep_fn(config.get("pretrained_model_name_or_path", "stabilityai/stable-diffusion-xl-base-1.0")),
            )
        else:
            if best_datapoint_per_round[previous_round]:
                if best_datapoint_per_round[previous_round].get("neighbors_improvement", False):
                    print("Using the best noise from the previous round")
                    prev_dp = best_datapoint_per_round[previous_round]
                    noises = {int(prev_dp["best_noise_seed"]): prev_dp["best_noise"]}
        
        # For zero-order, generate neighbors around the base noise
        if search_method == "zero-order":
            base_seed, base_noise = next(iter(noises.items()))
            neighbors = generate_neighbors(
                base_noise, threshold=search_args["threshold"], num_neighbors=search_args["num_neighbors"]
            ).squeeze(0)
            
            # Concatenate base noise with its neighbors
            neighbors_and_noise = torch.cat([base_noise, neighbors], dim=0)
            new_noises = {}
            for i, noise_tensor in enumerate(neighbors_and_noise):
                new_noises[base_seed + i] = noise_tensor.unsqueeze(0)
            noises = new_noises
        
        print(f"Number of noise samples: {len(noises)}")
        
        # Sample, verify, and save artifacts
        datapoint = sample(
            noises=noises,
            prompt=prompt,
            search_round=search_round,
            pipe=pipeline,
            verifier=verifier,
            topk=TOPK,
            root_dir=output_dir,
            config=config,
        )
        
        # Track results
        all_rounds_results.append(datapoint)
        
        # Update best result if needed
        if best_datapoint is None or datapoint["best_score"] > best_datapoint["best_score"]:
            best_datapoint = datapoint
        
        # For zero-order, update best datapoint per round
        if search_method == "zero-order":
            if datapoint.get("neighbors_improvement", False):
                best_datapoint_per_round[search_round] = datapoint
        
        search_round += 1
    
    # Compile final results
    final_results = {
        "prompt": prompt,
        "best_seed": int(best_datapoint['best_noise_seed']),
        "best_score": float(best_datapoint['best_score']),
        "best_round": int(best_datapoint['search_round']),
        "best_image_path": best_datapoint['best_img_path'],
        "search_method": search_method,
        "total_rounds": search_rounds,
        "all_rounds": all_rounds_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save final results
    results_file = os.path.join(output_dir, "final_results.json")
    with open(results_file, "w") as f:
        # Create a copy to avoid modifying the original
        save_results = {**final_results}
        # Remove non-serializable objects
        if "all_rounds" in save_results:
            # Remove noise tensors which aren't JSON serializable
            for round_data in save_results["all_rounds"]:
                if "best_noise" in round_data:
                    del round_data["best_noise"]
                if "all_results" in round_data:
                    for result in round_data["all_results"]:
                        if "noise" in result:
                            del result["noise"]
        
        json.dump(save_results, f, indent=4)
    
    print(f"\n===== SEARCH COMPLETE =====")
    print(f"Best seed found: {final_results['best_seed']} with score: {final_results['best_score']:.4f}")
    print(f"Found in round: {final_results['best_round']}")
    print(f"Best image path: {final_results['best_image_path']}")
    print(f"Results saved to {results_file}")
    
    # Return resources
    if return_pipe:
        final_results["pipeline"] = pipeline
        final_results["verifier"] = verifier
        
    # Free GPU memory if we created the pipeline
    elif not pipe_provided:
        del pipeline
        torch.cuda.empty_cache()
    
    return final_results


def parse_args():
    parser = argparse.ArgumentParser(description="Seed search for optimal image generation")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate images for")
    parser.add_argument("--config", type=str, default=None, help="Config file path (JSON)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", 
                        help="Model name or path")
    parser.add_argument("--lora_file", type=str, default=None, help="Optional LoRA file path")
    parser.add_argument("--search_rounds", type=int, default=3, help="Number of search rounds")
    parser.add_argument("--search_method", type=str, default="zero-order", 
                        choices=["zero-order", "random"], help="Search method to use")
    parser.add_argument("--verifier", type=str, default="gemini", 
                        choices=list(SUPPORTED_VERIFIERS.keys()), help="Verifier to use")
    parser.add_argument("--metric", type=str, default="overall_score", help="Metric to optimize for")
    parser.add_argument("--num_neighbors", type=int, default=8, help="Number of neighbors (zero-order only)")
    parser.add_argument("--threshold", type=float, default=0.01, help="Threshold for neighbor generation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    
    return parser.parse_args()


def main():
    """CLI entry point for seed search."""
    args = parse_args()
    
    # Load config file if provided, otherwise create from args
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = {
            "pretrained_model_name_or_path": args.model,
            "lora_file": args.lora_file,
            "torch_dtype": "float16",
            "use_low_gpu_vram": False,
            "batch_size_for_img_gen": args.batch_size,
            "search_args": {
                "search_method": args.search_method,
                "search_rounds": args.search_rounds,
                "threshold": args.threshold,
                "num_neighbors": args.num_neighbors,
            },
            "verifier_args": {
                "name": args.verifier,
                "choice_of_metric": args.metric,
            },
            "pipeline_call_args": {
                "height": 1024,
                "width": 1024,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            },
        }
    
    # Run the seed search
    results = run_seed_search(
        prompt=args.prompt,
        config=config,
        output_dir=args.output_dir
    )
    
    # Print final output
    print(f"\n🏆 BEST SEED: {results['best_seed']} with score {results['best_score']:.4f}")
    print(f"Image saved at: {results['best_image_path']}")
    
    return 0


if __name__ == "__main__":
    main()