"""
download_model.py
-----------------
Downloads best_model.pth from Hugging Face Hub at runtime.
Run this once locally to verify, then it runs automatically on deploy.

Usage:
    python download_model.py
"""

import os
import urllib.request

# ── Replace with your actual Hugging Face repo URL after uploading ──────────
# Format: https://huggingface.co/YOUR_HF_USERNAME/landwatch/resolve/main/best_model.pth
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/sudaise/landwatch/resolve/main/best_model.pth"
)

MODEL_PATH = "results/best_model.pth"


def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    os.makedirs("results", exist_ok=True)
    print(f"Downloading model from {MODEL_URL} ...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


if __name__ == "__main__":
    download_model()
