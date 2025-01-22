from datasets import load_dataset

repo_id = "data-is-better-together/open-image-preferences-v1-binarized"
ds = load_dataset(repo_id, split="train")
prompts = ds["prompt"]

with open("prompts_open_image_pref_v1.txt", "w") as f:
    for prompt in prompts:
        f.write(prompt.strip() + "\n")