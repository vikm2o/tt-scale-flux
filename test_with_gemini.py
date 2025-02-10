from google import genai
from google.genai import types
import typing_extensions as typing
from PIL import Image
import requests
import io
import json
import os


class Score(typing.TypedDict):
    score: float
    explanation: str


class Grading(typing.TypedDict):
    accuracy_to_prompt: Score
    creativity_and_originality: Score
    visual_quality_and_realism: Score
    consistency_and_cohesion: Score
    emotional_and_thematic_resonance: Score
    overall_score: Score


def load_image(path_or_url: str) -> Image.Image:
    """Load an image from a local path or a URL and return a PIL Image object."""
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url, stream=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    return Image.open(path_or_url)


def convert_to_bytes(path_or_url: str) -> bytes:
    """Load an image from a path or URL and convert it to bytes."""
    image = load_image(path_or_url).convert("RGB")
    image_bytes_io = io.BytesIO()
    image.save(image_bytes_io, format="PNG")
    return image_bytes_io.getvalue()


def prepare_inputs(prompt: str, image_path_or_uri: str):
    """Prepare inputs for the API from a given prompt and image."""
    inputs = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(data=convert_to_bytes(image_path_or_uri), mime_type="image/png"),
    ]
    return inputs


def load_verifier_prompt():
    """Loads the system prompt for Gemini when it acts as a verifier to grade images."""
    with open("verifier_prompt.txt", "r") as f:
        verifier_prompt = f.read().replace('"""', "")

    return verifier_prompt


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
system_instruction = load_verifier_prompt()
generation_config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    response_mime_type="application/json",
    response_schema=list[Grading],
    seed=1994,
)

# Define inputs
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

inputs = []
for text, path_or_url in image_urls:
    inputs.extend(prepare_inputs(prompt=text, image_path_or_uri=path_or_url))

# # Single image
# response = client.models.generate_content(
#     model='gemini-2.0-flash',
#     contents=[
#         "realistic photo a shiny black SUV car with a mountain in the background.",
#         load_image("https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/flux-edit-artifacts/assets/car.jpg")
#     ],
#     config=generation_config
# )

response = client.models.generate_content(
    model="gemini-2.0-flash", contents=types.Content(parts=inputs, role="user"), config=generation_config
)

with open("results.json", "w") as f:
    json.dump(response.parsed, f)

print(json.dumps(response.parsed, indent=4))
