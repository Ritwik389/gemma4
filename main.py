import torch
from transformers import AutoModelForVideoClassification, AutoProcessor
import cv2
import numpy as np
import json
from pathlib import Path

# Setup device acceleration (use MPS on Apple Silicon if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model and processor
model = AutoModelForVideoClassification.from_pretrained("Nikeytas/videomae-crime-detector-maxdata-v1").to(device)
processor = AutoProcessor.from_pretrained("Nikeytas/videomae-crime-detector-maxdata-v1")

def check_motion_and_visibility(frames, motion_threshold=1.0, black_threshold=12.0):
    """
    Returns True if the segment has visible content and sufficient motion.
    `frames` is a list of numpy RGB arrays.
    """
    # Convert frames to grayscale for processing
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    # 1. Check if the video chunk is pitch black (low average intensity)
    mean_intensities = [g.mean() for g in grays]
    overall_mean = np.mean(mean_intensities)
    if overall_mean < black_threshold:
        return False  # Pitch black / cut feed / smoke blackout
        
    # 2. Check for motion (average pixel difference between consecutive frames)
    diffs = []
    for i in range(len(grays) - 1):
        diff = cv2.absdiff(grays[i], grays[i+1])
        diffs.append(diff.mean())
        
    mean_motion = np.mean(diffs)
    if mean_motion < motion_threshold:
        return False  # Static / no movement
        
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
    
    # Segment video into non-overlapping chunks of size `chunk_size`
    bit_index = 0
    for start_idx in range(0, T - chunk_size + 1, chunk_size):
        end_idx = start_idx + chunk_size
        window_frames = all_frames[start_idx:end_idx]
        
        # Apply motion and visibility filter
        if not check_motion_and_visibility(window_frames, motion_threshold=1.0, black_threshold=12.0):
            # Skip model execution for static/black frames and mark as Non-Violent (class 0)
            predicted_class = 0
            confidence = 1.0
        else:
            inputs = processor(window_frames, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
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
    # Filter intervals where predicted class is 1 (Violent Crime) and confidence > threshold
    intervals = [
        bit["frame_range"] 
        for bit in suspicious_bits 
        if bit["predicted_class"] == 1 and bit["confidence"] >= confidence_threshold
    ]
    
    if not intervals:
        print(f"No intervals meet the confidence threshold (>= {confidence_threshold}). Output video not created.")
        return
        
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
            
        # Check if the frame belongs to any suspicious interval
        is_suspicious = False
        for start, end in intervals:
            if start <= frame_idx <= end:
                is_suspicious = True
                break
                
        if is_suspicious:
            out.write(frame)
            written_frames += 1
            
        frame_idx += 1
        
    cap.release()
    out.release()
    print(f"[ok] Stitched video saved with {written_frames} frames to: {output_path}")

if __name__ == "__main__":
    video_path = "videos/anomalies/tests.mp4"
    output_video_path = "outputs_single/tests_test.mp4"
    
    # 1. Detect suspicious bits
    suspicious_bits = detect_suspicious_bits(
        video_path, chunk_size=16, confidence_threshold=0.5, fps_target=2.0
    )
    
    # Output the suspicious bits JSON to stdout
    print("\n--- Detected Bits ---")
    print(json.dumps(suspicious_bits))
    
    # 2. Stitch together frames with confidence >= 0.7
    print("\n--- Stitching Process ---")
    stitch_suspicious_clips(video_path, output_video_path, suspicious_bits, confidence_threshold=0.7)
