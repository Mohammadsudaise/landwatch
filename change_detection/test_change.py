import os
import random
import torch
from change_detection.change_detector import ChangeDetector

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_DIR = "data/raw/2750"

detector = ChangeDetector(
    model_path="results/best_model.pth",
    data_dir=DATA_DIR,
    device=device
)

# Pick two random images (simulate different years)
classes = detector.classes

class1 = random.choice(classes)
class2 = random.choice(classes)

img1 = random.choice(os.listdir(os.path.join(DATA_DIR, class1)))
img2 = random.choice(os.listdir(os.path.join(DATA_DIR, class2)))

path1 = os.path.join(DATA_DIR, class1, img1)
path2 = os.path.join(DATA_DIR, class2, img2)

pred1, pred2, changed = detector.detect_change(path1, path2)

print("Image 1 predicted as:", pred1)
print("Image 2 predicted as:", pred2)

if changed:
    print("Land use changed!")
else:
    print("No change detected.")