from google import genai
from google.genai import types
import typing_extensions as typing
import json
import os
import sys
from typing import Union
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

sys.path.insert(0, current_dir)
sys.path.insert(0, root_dir)


from base_verifier import BaseVerifier
from utils import convert_to_bytes
from utils import image_to_base64


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


class GeminiVerifier(BaseVerifier):
    SUPPORTED_METRIC_CHOICES = [
        "accuracy_to_prompt",
        "creativity_and_originality",
        "visual_quality_and_realism",
        "consistency_and_cohesion",
        "emotional_or_thematic_resonance",
        "anatomical_correctness",
        "overall_score",
    ]

    def __init__(self, seed=1994, model_name="gemini-2.0-flash", **kwargs):
        # Remove unused sample_file parameter
        super().__init__(seed=seed, prompt_path=kwargs.pop("prompt_path", None))
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.sample_files = kwargs.pop("sample_files", None)
        
        print(f"Sample files: {self.sample_files}")

        # Keep system_instruction as text only
        self.system_instruction = self.verifier_prompt
        
        # Store example content separately
        self.example_content = None
        if self.sample_files is not None:
            example_parts = []
            # Create parts for the examples
            for sample_file in self.sample_files:
                example_parts.append(types.Part.from_bytes(
                    data=convert_to_bytes(sample_file), 
                    mime_type="image/png"
                ))
            # Add explanatory text
            example_parts.append(types.Part.from_text("These are example images to check for anatomical correctness."))
            self.example_content = types.Content(parts=example_parts, role="user")

        self.generation_config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            response_mime_type="application/json",
            response_schema=list[Grading],
            max_output_tokens=kwargs.pop("max_new_tokens", None),
            seed=seed,
        )
        self.model_name = model_name

    def prepare_inputs(self, images: Union[list[Image.Image], Image.Image], prompts: Union[list[str], str], **kwargs):
        """Prepare inputs for the API from a given prompt and image."""
        inputs = []
        images = images if isinstance(images, list) else [images]
        prompts = prompts if isinstance(prompts, list) else [prompts]
        for prompt, image in zip(prompts, images):
            part = [
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=convert_to_bytes(image), mime_type="image/png"),
            ]
            inputs.extend(part)

        return inputs

    def score(self, inputs, **kwargs) -> list[dict[str, float]]:
        def call_generate_content(parts):
            content = types.Content(parts=parts, role="user")
            
            # Create the conversation history with examples if available
            contents = []
            if self.example_content:
                contents.append(self.example_content)
                # Add model response acknowledging examples
                contents.append(types.Content(
                    parts=[types.Part.from_text("I'll use these examples as reference for my evaluation of anatomical correctness.")],
                    role="model"
                ))
            
            # Add the current content to evaluate
            contents.append(content)
            
            response = self.client.models.generate_content(
                model=self.model_name, 
                contents=contents,
                config=self.generation_config
            )
            return response.parsed[0]

        # Rest of the method remains the same
        grouped_inputs = [inputs[i : i + 2] for i in range(0, len(inputs), 2)]
        results = []
        max_workers = min(len(grouped_inputs), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_generate_content, group) for group in grouped_inputs]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"An error occurred during API call: {e}")
        return results


# Define inputs
if __name__ == "__main__":
    verifier = GeminiVerifier()
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
