import os
import json
import time
from pathlib import Path
import cv2
import numpy as np
import requests
import torch

from transformers import (
    AutoModelForMultimodalLM,
    AutoModelForVideoClassification,
    AutoProcessor,
)

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Telegram Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Load Video Classification Model (VideoMAE)
print("Loading VideoMAE model and processor...")
mae_model_id = "Nikeytas/videomae-crime-detector-maxdata-v1"
mae_model = AutoModelForVideoClassification.from_pretrained(mae_model_id).to(device)
mae_processor = AutoProcessor.from_pretrained(mae_model_id)

# Load Multimodal LLM (Gemma)
print("Loading Gemma multimodal model and processor...")
gemma_model_id = "google/gemma-4-E2B-it" 
gemma_model = AutoModelForMultimodalLM.from_pretrained(
    gemma_model_id,
    torch_dtype=torch.bfloat16,
).to(device)
gemma_processor = AutoProcessor.from_pretrained(gemma_model_id)


# ==========================================
# 2. TELEGRAM ALERT SYSTEM
# ==========================================
def send_telegram_alert(video_path, category, threat_level, description):
    """Sends the stitched anomaly video and Gemma-4's forensic JSON analysis to Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
    
    # Select threat color badge dynamically based on model output
    threat_emoji = "🔴" if threat_level.lower() == "high" else ("🟠" if threat_level.lower() == "medium" else "🟡")
    
    # Enhanced, highly scannable design format
    caption_text = (
        f"🚨 <b>CRITICAL SECURITY EVENT</b> 🚨\n"
        f"<code>──────────────────────────────</code>\n\n"
        f"🏷️ <b>INCIDENT INFORMATION</b>\n"
        f"• <b>Category:</b> <code>{category}</code>\n"
        f"• <b>Threat Level:</b> {threat_emoji} <b>{threat_level.upper()}</b>\n\n"
        f"🔬 <b>FORENSIC ANALYSIS REPORT</b>\n"
        f"<blockquote>{description}</blockquote>\n"
        f"<code>──────────────────────────────</code>\n"
    )
    
    print(f"Uploading {video_path} to Telegram... (Max limit: 50MB)")
    try:
        with open(video_path, "rb") as video_file:
            response = requests.post(
                url,
                data={"chat_id": CHAT_ID, "parse_mode": "HTML", "caption": caption_text},
                files={"video": video_file}
            )
        if response.status_code == 200:
            print("✓ Telegram alert sent successfully!")
        else:
            print(f"✗ Failed to send Telegram alert: {response.status_code}\n{response.text}")
    except FileNotFoundError:
        print(f"Error: Stitched file at {video_path} not found.")

# ==========================================
# 3. VIDEO PROCESSING FUNCTIONS
# ==========================================
def check_motion_and_visibility(frames, motion_threshold=1.0, black_threshold=12.0):
    """
    Returns True if the segment has visible content and sufficient motion.
    `frames` is a list of numpy RGB arrays.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    # 1. Check if the video chunk is pitch black
    mean_intensities = [g.mean() for g in grays]
    if np.mean(mean_intensities) < black_threshold:
        return False  
        
    # 2. Check for motion
    diffs = [cv2.absdiff(grays[i], grays[i+1]).mean() for i in range(len(grays) - 1)]
    if np.mean(diffs) < motion_threshold:
        return False  
        
    return True

def detect_suspicious_bits(video_path, chunk_size=16, confidence_threshold=0.5, fps_target=2.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
        
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps_target)))
    
    all_frames = []
    original_indices = []
    idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
            original_indices.append(idx)
        idx += 1
    cap.release()
    
    T = len(all_frames)
    if T < chunk_size:
        raise ValueError(f"Video has only {T} frames at {fps_target} FPS, but model requires at least {chunk_size} frames.")
        
    suspicious_bits = []
    bit_index = 0
    
    for start_idx in range(0, T - chunk_size + 1, chunk_size):
        end_idx = start_idx + chunk_size
        window_frames = all_frames[start_idx:end_idx]
        
        if not check_motion_and_visibility(window_frames, motion_threshold=1.0, black_threshold=12.0):
            predicted_class, confidence = 0, 1.0
        else:
            inputs = mae_processor(window_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = mae_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
        suspicious_bits.append({
            "bit_index": bit_index,
            "frame_range": [original_indices[start_idx], original_indices[end_idx - 1]],
            "confidence": round(confidence, 3),
            "predicted_class": predicted_class
        })
        bit_index += 1
        
    return suspicious_bits

def stitch_suspicious_clips(video_path, output_path, suspicious_bits, confidence_threshold=0.7):
    intervals = [
        bit["frame_range"] 
        for bit in suspicious_bits 
        if bit["predicted_class"] == 1 and bit["confidence"] >= confidence_threshold
    ]
    
    if not intervals:
        print(f"No intervals meet the confidence threshold (>= {confidence_threshold}). Output video not created.")
        return False
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Stitching {len(intervals)} intervals into '{output_path}'...")
    
    frame_idx = 0
    written_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        is_suspicious = any(start <= frame_idx <= end for start, end in intervals)
        if is_suspicious:
            out.write(frame)
            written_frames += 1
            
        frame_idx += 1
        
    cap.release()
    out.release()
    
    if written_frames > 0:
        print(f"[ok] Stitched video saved with {written_frames} frames to: {output_path}")
        return True
    return False


# ==========================================
# 4. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    input_video_path = "videos/anomalies/tests.mp4"
    stitched_video_path = "outputs_single/tests_test.mp4"
    prompt_file = "prompt.txt"
    
    # --- PHASE 1: Detect and Stitch ---
    print("\n=== PHASE 1: Anomaly Detection ===")
    suspicious_bits = detect_suspicious_bits(
        input_video_path, chunk_size=16, confidence_threshold=0.5, fps_target=2.0
    )
    
    print("\n--- Detected Bits ---")
    print(json.dumps(suspicious_bits, indent=2))
    
    print("\n--- Stitching Process ---")
    video_created = stitch_suspicious_clips(
        input_video_path, stitched_video_path, suspicious_bits, confidence_threshold=0.7
    )
    
    # --- PHASE 2: Multimodal LLM Analysis ---
    if video_created:
        print("\n=== PHASE 2: LLM Analysis ===")
        
        # Read the prompt
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except FileNotFoundError:
            print(f"Error: Could not find '{prompt_file}'. Please create this file and add your text prompt.")
            exit(1)
            
        messages = [
            {
                'role': 'user',
                'content': [
                    {"type": "video", "video": stitched_video_path},
                    {'type': 'text', 'text': prompt_text}
                ]
            }
        ]
        
        t = time.time()
        print("Preparing inputs for Gemma...")
        inputs = gemma_processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        print(f"Input prep took {time.time()-t:.1f}s")
        
        t = time.time()
        print("Generating response...")
        outputs = gemma_model.generate(**inputs, max_new_tokens=128)
        print(f"Generation took {time.time()-t:.1f}s")
        
        # Decode only the newly generated tokens
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = gemma_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        response_text = response[0].strip()
        print("\n=== LLM Response ===")
        print(response_text)
        
        # --- PHASE 3: Telegram Alert Dispatch ---
        print("\n=== PHASE 3: Dispatching Telegram Notification ===")
        try:
            # Parse Gemma's output directly into a JSON dictionary
            alert_data = json.loads(response_text)
            
            send_telegram_alert(
                video_path=stitched_video_path,
                category=alert_data.get("anomaly_category", "Unusual Activity"),
                threat_level=alert_data.get("threat_level", "Medium"),
                description=alert_data.get("evidence_description", "No description provided.")
            )
        except json.JSONDecodeError:
            print("⚠️ Warning: LLM response wasn't a clean JSON string. Routing to standard text alert fallback...")
            send_telegram_alert(
                video_path=stitched_video_path,
                category="Anomalous Activity",
                threat_level="High",
                description=response_text[:400]  # Safe slice bounds
            )
            
    else:
        print("\nSkipping LLM analysis and Telegram alert because no anomalous segments were detected.")