MODEL_ID = "google/gemma-4-E2B-it" 
from transformers import AutoProcessor, AutoModelForMultimodalLM
import torch
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).to("mps")
import time

with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

t = time.time()
print("Loading processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

from transformers import AutoProcessor

messages = [
    {
        'role': 'user',
        'content': [
            {"type": "video", "video": "explode.mp4"},
            {'type': 'text', 'text': prompt}
        ]
    }
]

print("Preparing inputs...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

print(f"Input prep took {time.time()-t:.1f}s")
t = time.time()
print("Generating...")
outputs = model.generate(**inputs, max_new_tokens=128)
print(f"Generation took {time.time()-t:.1f}s")
generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(response[0])