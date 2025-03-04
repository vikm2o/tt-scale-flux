import os
import sys
import base64
from typing import Union, List
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)

from openai import OpenAI
from base_verifier import BaseVerifier
from utils import convert_to_bytes


class OpenAIVerifier(BaseVerifier):
    SUPPORTED_METRIC_CHOICES = [
        "accuracy_to_prompt",
        "creativity_and_originality",
        "visual_quality_and_realism",
        "consistency_and_cohesion",
        "emotional_or_thematic_resonance",
        "anatomical_correctness",
        "overall_score",
    ]

    def __init__(self, seed=1994, model_name="gpt-4o-2024-11-20", **kwargs):
        super().__init__(seed=seed, prompt_path=kwargs.pop("prompt_path", None))
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.sample_files = kwargs.pop("sample_files", None)
        self.example_images = []

        print(f"Sample files: {self.sample_files}")
        # Process example images if provided
        if self.sample_files is not None:
            for sample_file in self.sample_files:
                # Convert image to base64
                with open(sample_file, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                self.example_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                })
        
        self.system_message = self.verifier_prompt
        self.model_name = model_name
        self.max_tokens = kwargs.pop("max_new_tokens", 1024)
        self.temperature = kwargs.pop("temperature", 0.0)

    def prepare_inputs(self, images: Union[List[Image.Image], Image.Image], prompts: Union[List[str], str], **kwargs):
        inputs = []
        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]
        for prompt, image in zip(prompts, images):
            # Convert image to base64
            if isinstance(image, str):
                with open(image, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:
                import io
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}"
                    }
                }
            ]
            inputs.append(content)
        
        return inputs

    def score(self, inputs, **kwargs):
        def call_api(content):
            messages = [
                {"role": "system", "content": self.system_message},
            ]
            
            # Add example images if available
            if self.example_images:
                example_content = [
                    {"type": "text", "text": "These are example images to check for anatomical correctness:"}
                ]
                example_content.extend(self.example_images)
                messages.append({"role": "user", "content": example_content})
                messages.append({"role": "assistant", "content": "I'll use these examples as reference for my evaluation of anatomical correctness."})
            
            # Add current content to evaluate
            messages.append({"role": "user", "content": content})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            try:
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"Error parsing OpenAI response: {e}")
                return None

        results = []
        max_workers = min(len(inputs), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_api, input_content) for input_content in inputs]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"An error occurred during API call: {e}")
        
        return results

# Define inputs
if __name__ == "__main__":
    verifier = OpenAIVerifier()
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

    # # Single image
    # response = client.models.generate_content(
    #     model='gemini-2.0-flash',
    #     contents=[
    #         "realistic photo a shiny black SUV car with a mountain in the background.",
    #         load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg")
    #     ],
    #     config=generation_config
    # )
    inputs = verifier.prepare_inputs(images=images, prompts=prompts)
    response = verifier.score(inputs)

    with open("results.json", "w") as f:
        json.dump(response, f)

    print(json.dumps(response, indent=4))
