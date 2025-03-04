import os
import json
import torch
import gradio as gr
from diffusers import DiffusionPipeline
from PIL import Image
import numpy as np

from seed_search import run_seed_search
from utils import get_noises, get_latent_prep_fn, TORCH_DTYPE_MAP
from verifiers import SUPPORTED_VERIFIERS

# Define available models
AVAILABLE_MODELS = {
    "Stable Diffusion XL": "stabilityai/stable-diffusion-xl-base-1.0",
    "Stable Diffusion 1.5": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
    "Dreamshaper XL": "dreamshaper-xl-1-0",
}

# Available verifiers
AVAILABLE_VERIFIERS = list(SUPPORTED_VERIFIERS.keys())

# Default metrics for all verifiers
DEFAULT_METRICS = {
    "accuracy_to_prompt": "How well the image matches the prompt",
    "creativity_and_originality": "Level of creativity and uniqueness",
    "visual_quality_and_realism": "Overall visual quality and realism",
    "consistency_and_cohesion": "Internal consistency of the image",
    "emotional_or_thematic_resonance": "Emotional impact of the image",
    "anatomical_correctness": "Correctness of body proportions",
    "overall_score": "Combined overall quality score"
}

# Cache for models
model_cache = {}

def generate_with_seed(prompt, model_name, seed, guidance_scale=7.5, steps=30, height=1024, width=1024, progress=gr.Progress()):
    """Generate an image using a specific seed"""
    # Use cached model or load new one
    cache_key = model_name
    if cache_key in model_cache:
        pipe = model_cache[cache_key]
    else:
        progress(0.1, desc="Loading model...")
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        pipe = pipe.to("cuda:0")
        model_cache[cache_key] = pipe
    
    # Create latents from seed
    progress(0.3, desc="Generating noise from seed...")
    latents = get_noises(
        max_seed=seed, 
        num_samples=1,
        height=height,
        width=width,
        dtype=torch.float16,
        fn=get_latent_prep_fn(model_name),
        fixed_seed=seed
    )
    latent = list(latents.values())[0]
    
    # Generate image
    progress(0.5, desc="Generating image...")
    result = pipe(
        prompt=prompt,
        latents=latent,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
    )
    
    progress(1.0, desc="Done!")
    return result.images[0]

def search_and_generate(prompt, model_name, verifier_name, metric, search_rounds, 
                        neighbors, threshold, guidance_scale, steps, height, width, status_box, progress=gr.Progress()):
    """Run seed search and generate final image with best seed"""
    status_box = "Starting seed search process...\n"
    progress(0, desc="Initializing...")
    
    # Prepare configuration
    config = {
        "pretrained_model_name_or_path": model_name,
        "torch_dtype": "float16",
        "use_low_gpu_vram": False,
        "batch_size_for_img_gen": 1,
        "search_args": {
            "search_method": "zero-order",
            "search_rounds": search_rounds,
            "threshold": threshold,
            "num_neighbors": neighbors,
        },
        "verifier_args": {
            "name": verifier_name,
            "choice_of_metric": metric,
        },
        "pipeline_call_args": {
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        },
    }
    
    # Create output directory
    output_dir = os.path.join("gradio_outputs", f"search_{os.urandom(4).hex()}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Custom status update function
    def status_update(message):
        nonlocal status_box
        status_box += f"{message}\n"
        return status_box
    
    status_update(f"Configuration prepared. Using {verifier_name} verifier with {metric} metric.")
    status_update(f"Running {search_rounds} search rounds with {neighbors} neighbors per round.")
    
    # Patch print function to update status
    original_print = print
    def custom_print(*args, **kwargs):
        message = " ".join(map(str, args))
        status_box = status_update(message)
        original_print(*args, **kwargs)
        return status_box
    
    import builtins
    builtins.print = custom_print
    
    try:
        # Run seed search
        progress(0.2, desc="Running seed search...")
        results = run_seed_search(
            prompt=prompt,
            config=config,
            output_dir=output_dir,
            return_pipe=True  # Return the pipeline to reuse it
        )
        
        # Extract results
        best_seed = results["best_seed"]
        best_score = results["best_score"]
        best_round = results["best_round"]
        best_image_path = results["best_image_path"]
        
        # Generate final image with best seed and pipeline
        progress(0.7, desc=f"Generating final image with seed {best_seed}...")
        status_update(f"✨ Best seed found: {best_seed} with score: {best_score:.4f} in round {best_round}")
        status_update(f"Generating final high-quality image with seed {best_seed}...")
        
        pipe = results.get("pipeline")
        final_image = generate_with_seed(
            prompt=prompt,
            model_name=model_name, 
            seed=best_seed,
            guidance_scale=guidance_scale,
            steps=steps,
            height=height,
            width=width,
            progress=progress
        )
        
        # Clean up
        if "pipeline" in results:
            del results["pipeline"]
        if "verifier" in results:
            del results["verifier"]
        torch.cuda.empty_cache()
        
        # Generate final status
        status_update("✅ Process complete!")
        status_update(f"Best seed: {best_seed}")
        status_update(f"Score: {best_score:.4f}")
        status_update(f"Found in round: {best_round}")
        
        # Try to load the best image from search for comparison
        search_image = None
        try:
            search_image = Image.open(best_image_path)
        except:
            status_update("Could not load search result image for comparison")
        
        # Reset print function
        builtins.print = original_print
        
        # Return the results
        if search_image:
            gallery = [search_image, final_image]
            return gallery, status_box, str(best_seed), f"{best_score:.4f}"
        else:
            return [final_image], status_box, str(best_seed), f"{best_score:.4f}"
            
    except Exception as e:
        status_update(f"❌ Error: {str(e)}")
        # Reset print function
        builtins.print = original_print
        return None, status_box, "Error", "Error"

# Create the Gradio interface
with gr.Blocks(title="Seed Search & Optimal Image Generator") as app:
    gr.Markdown("# 🔍 Seed Search & Optimal Image Generator")
    gr.Markdown("Find the best seed for your prompt and generate high-quality images")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Inputs
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your image generation prompt here...",
                lines=3
            )
            
            with gr.Accordion("Model Settings", open=True):
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value=list(AVAILABLE_MODELS.keys())[0],
                    label="Select Model"
                )
                
                verifier_dropdown = gr.Dropdown(
                    choices=AVAILABLE_VERIFIERS,
                    value="gemini" if "gemini" in AVAILABLE_VERIFIERS else AVAILABLE_VERIFIERS[0],
                    label="Select Verifier"
                )
                
                metric_dropdown = gr.Dropdown(
                    choices=list(DEFAULT_METRICS.keys()),
                    value="overall_score",
                    label="Select Metric"
                )
            
            with gr.Accordion("Search Settings", open=True):
                search_rounds = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Search Rounds"
                )
                
                neighbors = gr.Slider(
                    minimum=4, maximum=16, value=8, step=2,
                    label="Neighbors per Round"
                )
                
                threshold = gr.Slider(
                    minimum=0.005, maximum=0.05, value=0.01, step=0.005,
                    label="Noise Threshold"
                )
            
            with gr.Accordion("Image Settings", open=False):
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=15.0, value=7.5, step=0.5,
                    label="Guidance Scale"
                )
                
                steps = gr.Slider(
                    minimum=20, maximum=100, value=30, step=5,
                    label="Steps"
                )
                
                with gr.Row():
                    height = gr.Dropdown(
                        choices=[512, 768, 1024, 1280], 
                        value=1024,
                        label="Height"
                    )
                    width = gr.Dropdown(
                        choices=[512, 768, 1024, 1280],
                        value=1024,
                        label="Width"
                    )
            
            # Action button
            generate_btn = gr.Button("Find Best Seed & Generate", variant="primary")
        
        with gr.Column(scale=1):
            # Outputs
            gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery",
                columns=2,
                height=512
            )
            
            status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=10
            )
            
            with gr.Row():
                best_seed = gr.Textbox(label="Best Seed", interactive=False)
                best_score = gr.Textbox(label="Score", interactive=False)
    
    # Set up event handlers
    generate_btn.click(
        search_and_generate,
        inputs=[
            prompt_input,
            lambda model: AVAILABLE_MODELS[model],  # Convert selection to model path
            verifier_dropdown,
            metric_dropdown,
            search_rounds,
            neighbors,
            threshold,
            guidance_scale,
            steps,
            height,
            width,
            status
        ],
        outputs=[gallery, status, best_seed, best_score]
    )

    # Provide examples
    gr.Examples(
        examples=[
            ["A majestic lion standing on a cliff at sunset"],
            ["A cozy cabin in a snowy forest with smoke coming from the chimney"],
            ["An astronaut riding a horse on Mars, photorealistic"],
        ],
        inputs=prompt_input
    )

    gr.Markdown("### How it works")
    gr.Markdown("""
    1. **Seed Search**: The app runs multiple rounds of zero-order optimization to find the best seed.
    2. **Evaluation**: Each image is evaluated by the selected verifier based on the chosen metric.
    3. **Final Generation**: Once the best seed is found, a high-quality image is generated.
    
    The gallery shows the best seed search result (left) and the final generated image (right).
    """)

# Launch the app
if __name__ == "__main__":
    app.launch()