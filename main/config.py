import os
import torch
from dotenv import load_dotenv
load_dotenv()

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

MAE_MODEL_ID = "Nikeytas/videomae-crime-detector-maxdata-v1"
GEMMA_MODEL_ID = "google/gemma-4-E2B-it"  