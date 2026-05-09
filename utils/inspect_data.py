import os
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = "data/raw/2750"

classes = os.listdir(DATA_DIR)
print("Classes:", classes)

cls = random.choice(classes)
img_name = random.choice(os.listdir(os.path.join(DATA_DIR, cls)))
img_path = os.path.join(DATA_DIR, cls, img_name)

img = Image.open(img_path)
plt.imshow(img)
plt.title(f"Class: {cls}")
plt.axis("off")
plt.show()
