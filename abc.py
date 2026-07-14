import torch
from transformers import AutoModelForVideoClassification, AutoProcessor
import cv2
import numpy as np

# Load model and processor
model = AutoModelForVideoClassification.from_pretrained("Nikeytas/videomae-crime-detector-maxdata-v1")
processor = AutoProcessor.from_pretrained("Nikeytas/videomae-crime-detector-maxdata-v1")

# Process video
def classify_video(video_path, num_frames=16):
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    # Process with model
    inputs = processor(frames, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    label = "Violent Crime" if predicted_class == 1 else "Non-Violent"
    return label, confidence

# Example usage
video_path = "abc.mp4"
prediction, confidence = classify_video(video_path)
print(f"Prediction: {prediction} (Confidence: {confidence:.3f})")
