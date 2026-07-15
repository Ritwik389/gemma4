import os
import requests
import cv2
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# 2. Define the enhanced alert function
def send_video_alert(video_path, result):
    """
    Sends a video file to a Telegram chat with a cleanly formatted HTML caption.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"
    
    caption_text = (
        f"🚨 <b>CRITICAL ALERT DETECTED</b>\n"
        f"⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯\n"
        f"<b>Event Class:</b> <code>{result['class']}</code>\n\n"
        f"<b>Description:</b>\n"
        f"<blockquote>{result['description']}</blockquote>"
    )
    
    print(f"Uploading {video_path} to Telegram... (Max limit: 50MB)")
    
    try:
        with open(video_path, "rb") as video_file:
            response = requests.post(
                url,
                data={
                    "chat_id": CHAT_ID,
                    "parse_mode": "HTML",
                    "caption": caption_text
                },
                files={
                    "video": video_file
                }
            )
            
        # Check if the request was successful
        if response.status_code == 200:
            print("Alert sent successfully!")
        else:
            print(f"Failed to send alert. Server responded with: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print(f"Error: The file at {video_path} could not be found.")

# 3. Execution logic
if __name__ == "__main__":
    # Mock analysis result (replace this with your actual model output)
    result = {
        "class": "Explosion",
        "description": "Bright flash followed by rapid expansion of fire and structural debris."
    }
    
    video_path = "final_videos/Explosion.mp4"
    
    # Trigger condition
    if result["class"] == "Explosion":
        send_video_alert(video_path, result)