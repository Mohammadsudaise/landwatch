import torch
import os
import random
import matplotlib.pyplot as plt
from change_detection.spatial_change import SpatialChangeDetector

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_DIR = "data/raw/2750"

detector = SpatialChangeDetector(
    model_path="results/best_model.pth",
    data_dir=DATA_DIR,
    device=device
)

classes = detector.classes

class1 = random.choice(classes)
class2 = random.choice(classes)

img1 = random.choice(os.listdir(os.path.join(DATA_DIR, class1)))
img2 = random.choice(os.listdir(os.path.join(DATA_DIR, class2)))

path1 = os.path.join(DATA_DIR, class1, img1)
path2 = os.path.join(DATA_DIR, class2, img2)

image1, image2, change_map, change_percentage = detector.detect_spatial_change(path1, path2)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image1)
plt.title("Image 1")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(image2)
plt.title("Image 2")
plt.axis("off")

plt.subplot(1,3,3)

# Overlay change map on image
plt.imshow(image2)
plt.imshow(change_map, cmap="jet", alpha=0.5)

plt.title("Overlay Change Map")
plt.axis("off")
print(f"Land change detected: {change_percentage:.2f}%")
plt.show()