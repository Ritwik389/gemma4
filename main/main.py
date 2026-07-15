# main.py
import json
from video_mae import detect_suspicious_bits, stitch_suspicious_clips
from gemma_llm import analyze_video
from telegram_alert import send_telegram_alert

def main():
    input_video_path = "final_videos/Explosion.mp4"
    stitched_video_path = "outputs_single/tests_test.mp4"
    prompt_file = "prompt.txt"
    
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find '{prompt_file}'. Creating a default task prompt template...")
        prompt_text = (
            "Review this security camera recording and return a JSON containing "
            "the keys 'anomaly_category' and 'evidence_description'."
        )
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt_text)

    # ==========================================
    # PHASE 1: Anomaly Detection & Stitching
    # ==========================================
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
    
    # ==========================================
    # PHASE 2: Multimodal LLM Analysis
    # ==========================================
    if video_created:
        print("\n=== PHASE 2: LLM Analysis ===")
        response_text = analyze_video(stitched_video_path, prompt_text)
        
        print("\n=== LLM Response ===")
        print(response_text)
        
        # ==========================================
        # PHASE 3: Telegram Alert Dispatch
        # ==========================================
        print("\n=== PHASE 3: Dispatching Telegram Notification ===")
        try:
            # Parse Gemma's output directly into a JSON dictionary
            alert_data = json.loads(response_text)
            
            send_telegram_alert(
                video_path=stitched_video_path,
                category=alert_data.get("anomaly_category", "Unusual Activity"),
                description=alert_data.get("evidence_description", "No description provided.")
            )
        except json.JSONDecodeError:
            print("⚠️ Warning: LLM response wasn't clean JSON. Routing to fallback alert...")
            send_telegram_alert(
                video_path=stitched_video_path,
                category="Anomalous Activity",
                description=response_text[:400]  
            )
    else:
        print("\nSkipping LLM analysis and Telegram alerts because no anomalous segments were detected.")

if __name__ == "__main__":
    main()