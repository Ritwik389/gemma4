import time
import av
import numpy as np
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
from config import DEVICE, GEMMA_MODEL_ID

_gemma_model = None
_gemma_processor = None

def get_gemma_model():
    global _gemma_model, _gemma_processor
    if _gemma_model is None:
        print("Loading Gemma multimodal model and processor...")
        _gemma_model = AutoModelForMultimodalLM.from_pretrained(
            GEMMA_MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
        _gemma_processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID)
    return _gemma_model, _gemma_processor


def analyze_video(stitched_video_path, prompt_text):
    gemma_model, gemma_processor = get_gemma_model()
    
    print("Decoding stitched video frames via PyAV...")
    container = av.open(stitched_video_path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_rgb().to_ndarray())
    container.close()
    video_frames = np.stack(frames)
    
    messages = [
        {
            'role': 'user',
            'content': [
                {"type": "video"},  
                {'type': 'text', 'text': prompt_text}
            ]
        }
    ]
    
    t = time.time()
    print("Preparing inputs for Gemma...")
    formatted_prompt = gemma_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = gemma_processor(
        text=formatted_prompt,
        videos=video_frames,
        return_tensors="pt"
    ).to(gemma_model.device)
    print(f"Input prep took {time.time()-t:.1f}s")
    
    t = time.time()
    print("Generating response...")
    with torch.no_grad():
        outputs = gemma_model.generate(
            **inputs, 
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False
        )
    print(f"Generation took {time.time()-t:.1f}s")
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = gemma_processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response_text