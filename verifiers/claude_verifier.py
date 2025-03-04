import os
import sys
import json
import base64
import typing_extensions as typing
from typing import Union, List, Dict, Any
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)

from base_verifier import BaseVerifier
from utils import convert_to_bytes, image_to_base64


class Score(typing.TypedDict):
    explanation: str
    score: float


class Grading(typing.TypedDict):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_or_thematic_resonance: Score
    anatomical_correctness: Score
    overall_score: Score


class ClaudeVerifier(BaseVerifier):
    SUPPORTED_METRIC_CHOICES = [
        "accuracy_to_prompt",
        "creativity_and_originality",
        "visual_quality_and_realism",
        "consistency_and_cohesion",
        "emotional_or_thematic_resonance",
        "anatomical_correctness",
        "overall_score",
    ]

    def __init__(self, seed=1994, model_name="claude-3-7-sonnet-20240229", **kwargs):
        super().__init__(seed=seed, prompt_path=kwargs.pop("prompt_path", None))
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.sample_files = kwargs.pop("sample_files", None)

        print(f"Sample files: {self.sample_files}")
        # Set system prompt
        self.system_prompt = self.verifier_prompt
        
        # Store example content separately
        self.example_content = None
        if self.sample_files is not None:
            # Build example content message with images
            self.example_images = []
            for sample_file in self.sample_files:
                # Read the image file
                with open(sample_file, "rb") as f:
                    image_data = f.read()
                # Convert to base64
                image_b64 = base64.b64encode(image_data).decode("utf-8")
                self.example_images.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64
                    }
                })
        
        self.model_name = model_name
        self.max_tokens = kwargs.pop("max_new_tokens", 4096)
        self.temperature = kwargs.pop("temperature", 0.0)
    def parse_json(self,s):
        s = s[next(idx for idx, c in enumerate(s) if c in "{["):]
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            return json.loads(s[:e.pos])
    
    def prepare_inputs(self, images: Union[List[Image.Image], Image.Image], prompts: Union[List[str], str], **kwargs):
        """Prepare inputs for the API from given prompts and images."""
        inputs = []
        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]
        
        for prompt, image in zip(prompts, images):
            # Convert image to base64
            if isinstance(image, str):  # If it's a file path
                with open(image, "rb") as img_file:
                    img_data = img_file.read()
                img_b64 = base64.b64encode(img_data).decode("utf-8")
            else:  # If it's a PIL Image
                import io
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Create message content
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64
                    }
                }
            ]
            inputs.append(content)
        
        return inputs

    def score(self, inputs, **kwargs) -> list[dict[str, float]]:
        def call_claude_api(content):
            messages = []
            
            # Add examples if available
            if self.example_images:
                example_content = []
                # Add images
                for img in self.example_images:
                    example_content.append(img)
                # Add text
                example_content.append({"type": "text", "text": "These are example images to check for anatomical correctness."})
                
                messages.append({"role": "user", "content": example_content})
                messages.append({
                    "role": "assistant", 
                    "content": "I'll use these examples as reference for my evaluation of anatomical correctness."
                })
            
            # Add current content to evaluate
            messages.append({"role": "user", "content": content})
            
            # Modify system prompt to request JSON format
            system_prompt = self.system_prompt
            if not "JSON" in system_prompt and not "json" in system_prompt:
                system_prompt += "\nPlease format your response as a valid JSON object."
            
            try:
                # Try with different parameter options based on SDK version
                try:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        system=system_prompt,
                        messages=messages,
                        temperature=self.temperature,
                        # Remove response_format parameter that's causing the error
                    )
                except TypeError:
                    # If that fails, try an alternative approach for older SDK versions
                    import anthropic
                    # Print SDK version for debugging
                    print(f"Using Anthropic SDK version: {anthropic.__version__}")
                    
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        system=system_prompt,
                        messages=messages,
                        temperature=self.temperature,
                    )
                
                # Parse the JSON response
                try:
                    json_response = json.loads(response.content[0].text)
                    return json_response
                except Exception as e:
                    print(f"Error parsing Claude response: {e}")
                    print(f"Raw response: {response.content[0].text}")
                    return None
                    
            except Exception as e:
                print(f"Error calling Claude API: {e}")
                return None

        results = []
        max_workers = min(len(inputs), 4)  # Limit concurrency
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_claude_api, input_content) for input_content in inputs]
            for future in as_completed(futures):
                try:
                    result = self.parse_json(future.result())
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"An error occurred during API call: {e}")
        
        return results


# Define inputs
if __name__ == "__main__":
    verifier = ClaudeVerifier()
    image_urls = [
        (
            "realistic photo a shiny black SUV car with a mountain in the background.",
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg",
        ),
        (
            "photo a green and funny creature standing in front a lightweight forest.",
            "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/green_creature.jpg",
        ),
    ]

    prompts = []
    images = []
    for text, path_or_url in image_urls:
        prompts.append(text)
        images.append(path_or_url)

    inputs = verifier.prepare_inputs(images=images, prompts=prompts)
    response = verifier.score(inputs)

    with open("claude_results.json", "w") as f:
        json.dump(response, f)

    print(json.dumps(response, indent=4))