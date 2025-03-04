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

    def __init__(self, seed=1994, model_name="claude-3-7-sonnet-20250219", **kwargs):
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
            
            # Make system prompt explicitly request JSON with specific format
            system_prompt = self.system_prompt
            system_prompt += "\n\nIMPORTANT: Your response MUST be a valid JSON object matching this structure:\n"
            system_prompt += '''{
                "accuracy_to_prompt": {"explanation": "string", "score": number},
                "creativity_and_originality": {"explanation": "string", "score": number},
                "visual_quality_and_realism": {"explanation": "string", "score": number},
                "consistency_and_cohesion": {"explanation": "string", "score": number},
                "emotional_or_thematic_resonance": {"explanation": "string", "score": number},
                "anatomical_correctness": {"explanation": "string", "score": number},
                "overall_score": {"explanation": "string", "score": number}
            }'''
            
            try:
                # Get SDK version
                import anthropic
                sdk_version = anthropic.__version__
                print(f"Using Anthropic SDK version: {sdk_version}")
                
                # Different API call based on SDK version
                if hasattr(self.client.messages, 'create'):
                    # For newer API
                    try:
                        # Parse version more carefully
                        version_parts = sdk_version.split('.')
                        major, minor = int(version_parts[0]), int(version_parts[1])
                        
                        if major > 0 or (major == 0 and minor >= 5):
                            # Use 'format' parameter for SDK 0.5.0+
                            response = self.client.messages.create(
                                model=self.model_name,
                                max_tokens=self.max_tokens,
                                system=system_prompt,
                                messages=messages,
                                temperature=self.temperature
                            )
                        else:
                            # For versions like 0.49.0, don't use 'format' parameter
                            print(f"Using SDK version {sdk_version} without format parameter")
                            response = self.client.messages.create(
                                model=self.model_name,
                                max_tokens=self.max_tokens,
                                system=system_prompt,
                                messages=messages,
                                temperature=self.temperature
                            )
                    except Exception as e:
                        print(f"Error with API call: {e}")
                        # Last resort fallback
                        response = self.client.messages.create(
                            model=self.model_name,
                            max_tokens=self.max_tokens,
                            system=system_prompt,
                            messages=messages,
                            temperature=self.temperature
                        )
                else:
                    # For even older SDK versions
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        system=system_prompt,
                        messages=messages,
                        temperature=self.temperature
                    )
                    
                # Parse response
                if hasattr(response, 'content') and response.content:
                    content_text = response.content[0].text.strip()
                    print(f"Raw response text (first 100 chars): {content_text[:100]}...")
                    
                    # Try to extract JSON from text (some models add markdown code blocks)
                    import re
                    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                    json_match = re.search(json_pattern, content_text)
                    
                    if json_match:
                        # Extract JSON from code block
                        json_text = json_match.group(1).strip()
                        print("Found JSON in code block")
                    else:
                        # Try to use the whole text
                        json_text = content_text
                    
                    # Try parsing the JSON
                    try:
                        json_response = json.loads(json_text)
                        print("Successfully parsed JSON")
                        return json_response
                    except json.JSONDecodeError as e:
                        print(f"JSON error: {e}")
                        # As a fallback, create a basic structure with the raw text
                        return {
                            "accuracy_to_prompt": {"explanation": "Error parsing response", "score": 0.0},
                            "creativity_and_originality": {"explanation": "Error parsing response", "score": 0.0},
                            "visual_quality_and_realism": {"explanation": "Error parsing response", "score": 0.0},
                            "consistency_and_cohesion": {"explanation": "Error parsing response", "score": 0.0},
                            "emotional_or_thematic_resonance": {"explanation": "Error parsing response", "score": 0.0},
                            "anatomical_correctness": {"explanation": "Error parsing response", "score": 0.0},
                            "overall_score": {"explanation": "Error parsing response", "score": 0.0},
                            "raw_text": content_text[:500]  # Include some of the raw text
                        }
                else:
                    print("Response has no content")
                    # Return empty structure
                    return {
                        "accuracy_to_prompt": {"explanation": "No response received", "score": 0.0},
                        "creativity_and_originality": {"explanation": "No response received", "score": 0.0},
                        "visual_quality_and_realism": {"explanation": "No response received", "score": 0.0},
                        "consistency_and_cohesion": {"explanation": "No response received", "score": 0.0},
                        "emotional_or_thematic_resonance": {"explanation": "No response received", "score": 0.0},
                        "anatomical_correctness": {"explanation": "No response received", "score": 0.0},
                        "overall_score": {"explanation": "No response received", "score": 0.0}
                    }
                    
            except Exception as e:
                print(f"Error calling Claude API: {e}")
                # Return empty structure
                return {
                    "accuracy_to_prompt": {"explanation": f"API error: {str(e)}", "score": 0.0},
                    "creativity_and_originality": {"explanation": f"API error: {str(e)}", "score": 0.0},
                    "visual_quality_and_realism": {"explanation": f"API error: {str(e)}", "score": 0.0},
                    "consistency_and_cohesion": {"explanation": f"API error: {str(e)}", "score": 0.0},
                    "emotional_or_thematic_resonance": {"explanation": f"API error: {str(e)}", "score": 0.0},
                    "anatomical_correctness": {"explanation": f"API error: {str(e)}", "score": 0.0},
                    "overall_score": {"explanation": f"API error: {str(e)}", "score": 0.0}
                }

        results = []
        max_workers = min(len(inputs), 4)  # Limit concurrency
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_claude_api, input_content) for input_content in inputs]
            for future in as_completed(futures):
                try:
                    result = future.result() # Get the result
                    print(f"Result : {(result)}")
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"An error occurred processing future: {e}")
                    # Add a fallback result
                    results.append({
                        "accuracy_to_prompt": {"explanation": f"Processing error: {str(e)}", "score": 0.0},
                        "creativity_and_originality": {"explanation": f"Processing error: {str(e)}", "score": 0.0},
                        "visual_quality_and_realism": {"explanation": f"Processing error: {str(e)}", "score": 0.0},
                        "consistency_and_cohesion": {"explanation": f"Processing error: {str(e)}", "score": 0.0},
                        "emotional_or_thematic_resonance": {"explanation": f"Processing error: {str(e)}", "score": 0.0},
                        "anatomical_correctness": {"explanation": f"Processing error: {str(e)}", "score": 0.0},
                        "overall_score": {"explanation": f"Processing error: {str(e)}", "score": 0.0}
                    })
        
        # Ensure we return the same number of results as inputs
        if len(results) < len(inputs):
            for _ in range(len(inputs) - len(results)):
                results.append({
                    "accuracy_to_prompt": {"explanation": "Missing result", "score": 0.0},
                    "creativity_and_originality": {"explanation": "Missing result", "score": 0.0},
                    "visual_quality_and_realism": {"explanation": "Missing result", "score": 0.0},
                    "consistency_and_cohesion": {"explanation": "Missing result", "score": 0.0},
                    "emotional_or_thematic_resonance": {"explanation": "Missing result", "score": 0.0},
                    "anatomical_correctness": {"explanation": "Missing result", "score": 0.0},
                    "overall_score": {"explanation": "Missing result", "score": 0.0}
                })
        
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