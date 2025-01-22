from verifier import load_verifier, prepare_inputs, perform_inference
from utils import load_verifier_prompt, recover_json_from_output
from PIL import Image 
import os 

images = [Image.open(path) for path in sorted(os.listdir(".")) if path.endswith(".png")][:2]
verifier, processor = load_verifier()
verifier_prompt = load_verifier_prompt("verifier_prompt.txt")

prompt = "a bustling manga street, devoid of vehicles, detailed with vibrant colors and dynamic line work, characters in the background adding life and movement, under a soft golden hour light, with rich textures and a lively atmosphere, high resolution, sharp focus"
verifier_inputs = prepare_inputs(
    system_prompt=verifier_prompt, 
    images=images,
    prompts=[prompt] * len(images), 
    processor=processor
)
out = perform_inference(model=verifier, processor=processor, inputs=verifier_inputs, max_new_tokens=300)
print(len(out))
for o in out:
    print(recover_json_from_output(o))