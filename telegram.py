import cv2

video = cv2.VideoCapture("explode.mp4")

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
middle = frame_count // 2

video.set(cv2.CAP_PROP_POS_FRAMES, middle)

success, frame = video.read()

if success:
    cv2.imwrite("alert.jpg", frame)

video.release()


import requests
from dotenv import load_dotenv
import os

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_alert(image_path, result):

    with open(image_path, "rb") as photo:

        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
            data={
                "chat_id": CHAT_ID,
                "caption":
                    f"🚨 ALERT\n\n"
                    f"Class: {result['class']}\n\n"
                    f"{result['description']}"
            },
            files={
                "photo": photo
            }
        )

result = {
    "class": "Explosion",
    "description": "Bright flash followed by fire and debris..."
}

if result["class"] == "Explosion":
    send_alert("alert.jpg", result)