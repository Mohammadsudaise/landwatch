import torch
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
from preprocessing.transforms import get_transforms
from models.cnn_classifier import LandUseCNN

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load classes
DATA_DIR = "data/raw/2750"
classes = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

# Load model
model = LandUseCNN(num_classes=len(classes))
model.load_state_dict(torch.load("results/best_model.pth"))
model.to(device)
model.eval()

# Pick random image
random_class = random.choice(classes)
img_name = random.choice(os.listdir(os.path.join(DATA_DIR, random_class)))
img_path = os.path.join(DATA_DIR, random_class, img_name)

# Preprocess image
transform = get_transforms()
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

predicted_class = classes[predicted.item()]

# Show result
plt.imshow(image)
plt.title(f"Actual: {random_class}\nPredicted: {predicted_class}")
plt.axis("off")
plt.show()