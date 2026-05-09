import torch
import matplotlib.pyplot as plt
from change_detection.spatial_change import SpatialChangeDetector

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

detector = SpatialChangeDetector(
    model_path="results/best_model.pth",
    data_dir="data/raw/2750",
    device=device
)

img1 = "real_satellite_data/image_2018.jpg"
img2 = "real_satellite_data/image_2024.jpg"

image1, image2, change_map, change_percentage = detector.detect_spatial_change(img1, img2)

print(f"Land change detected: {change_percentage:.2f}%")

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
plt.imshow(image2)
plt.imshow(change_map, cmap="jet", alpha=0.5)
plt.title("Change Map")
plt.axis("off")

plt.show()