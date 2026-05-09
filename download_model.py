import os
import urllib.request

MODEL_URL = "https://huggingface.co/sudaise/landwatch/resolve/main/best_model.pth?download=true"
MODEL_PATH = "results/best_model.pth"


def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    os.makedirs("results", exist_ok=True)
    print(f"Downloading model from Hugging Face...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


if __name__ == "__main__":
    download_model()
