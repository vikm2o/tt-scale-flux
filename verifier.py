from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch 

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE_MAP = {'visual': 1, 'model.embed_tokens': 1, 'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1, 'model.layers.11': "cpu", 'model.layers.12': "cpu", 'model.layers.13': "cpu", 'model.layers.14': "cpu", 'model.layers.15': "cpu", 'model.layers.16': "cpu", 'model.layers.17': "cpu", 'model.layers.18': "cpu", 'model.layers.19': "cpu", 'model.layers.20': "cpu", 'model.layers.21': "cpu", 'model.layers.22': "cpu", 'model.layers.23': "cpu", 'model.layers.24': "cpu", 'model.layers.25': "cpu", 'model.layers.26': "cpu", 'model.layers.27': "cpu", 'model.norm': "cpu", 'model.rotary_emb': "cpu", 'lm_head': "cpu"}


def load_verifier():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).to("cuda:1") # hard code for now.
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def prepare_conversations(system_prompt, prompt):
    user_content = []
    conversation = [
        {
            "role": "system",
            "content": system_prompt
        },
    ]
    user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": prompt})
    user_content = {"role": "user", "content": user_content}
    conversation.append(user_content)
    return conversation


def prepare_inputs(system_prompt: str, images: list, prompts: list, processor: AutoProcessor):
    assert len(images) == len(prompts)
    
    conversations = []
    for prompt in prompts:
        conversations.append(prepare_conversations(system_prompt, prompt))
    
    assert len(conversations) == len(images) == len(prompts)

    texts = [processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:1") # hard-code for now.
    return inputs

def perform_inference(model, processor, inputs, max_new_tokens):
    # TODO: might need to iterate `inputs` in batches depending on the resources.
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text