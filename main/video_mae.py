import cv2
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForVideoClassification, AutoProcessor
from config import DEVICE, MAE_MODEL_ID

_mae_model = None
_mae_processor = None

def get_mae_model():
    """Lazy loads VideoMAE to preserve system memory until called."""
    global _mae_model, _mae_processor
    if _mae_model is None:
        print("Loading VideoMAE model and processor...")
        _mae_model = AutoModelForVideoClassification.from_pretrained(MAE_MODEL_ID).to(DEVICE)
        _mae_processor = AutoProcessor.from_pretrained(MAE_MODEL_ID)
    return _mae_model, _mae_processor


def check_motion_and_visibility(frames, motion_threshold=1.0, black_threshold=12.0):
    """Returns True if the segment has visible content and sufficient motion."""
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    # 1. Check if the video chunk is pitch black
    mean_intensities = [g.mean() for g in grays]
    if np.mean(mean_intensities) < black_threshold:
        return False  
        
    # 2. Check for motion
    diffs = [cv2.absdiff(grays[i], grays[i+1]).mean() for i in range(len(grays) - 1)]
    return np.mean(diffs) >= motion_threshold


def detect_suspicious_bits(video_path, chunk_size=16, confidence_threshold=0.5, fps_target=2.0):
    """Processes video in blocks and scores anomalies with VideoMAE."""
    mae_model, mae_processor = get_mae_model()
    
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
        raise ValueError(f"Video has only {T} frames, but model requires at least {chunk_size} frames.")
        
    suspicious_bits = []
    bit_index = 0
    
    for start_idx in range(0, T - chunk_size + 1, chunk_size):
        end_idx = start_idx + chunk_size
        window_frames = all_frames[start_idx:end_idx]
        
        if not check_motion_and_visibility(window_frames):
            predicted_class, confidence = 0, 1.0
        else:
            inputs = mae_processor(window_frames, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
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
    """Stitches highlighted anomaly frames into a clean output clip."""
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