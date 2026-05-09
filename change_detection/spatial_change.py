import torch
import numpy as np
from PIL import Image
import os

from models.resnet_model import ResNetLandUse
from preprocessing.transforms import get_transforms

# Hardcoded EuroSAT classes — no need for data_dir on cloud
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


class SpatialChangeDetector:

    def __init__(self, model_path, data_dir=None, device=None):

        self.device = device or torch.device("cpu")

        # Use hardcoded classes — works on cloud where data_dir doesn't exist
        if data_dir and os.path.isdir(data_dir):
            self.classes = sorted([
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ])
        else:
            self.classes = EUROSAT_CLASSES

        # map_location="cpu" ensures it works on any server
        # even if the model was saved on MPS (Mac GPU)
        self.model = ResNetLandUse(num_classes=len(self.classes))
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_transforms()

    def predict_patch(self, patch):
        tensor = self.transform(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    def detect_spatial_change(self, img_path1, img_path2, patch_size=32):
        img1 = Image.open(img_path1).convert("RGB")
        img2 = Image.open(img_path2).convert("RGB")

        img1 = img1.resize((256, 256))
        img2 = img2.resize((256, 256))

        change_map = np.zeros((256, 256))

        for i in range(0, 256, patch_size):
            for j in range(0, 256, patch_size):
                patch1 = img1.crop((j, i, j + patch_size, i + patch_size))
                patch2 = img2.crop((j, i, j + patch_size, i + patch_size))

                pred1 = self.predict_patch(patch1)
                pred2 = self.predict_patch(patch2)

                if pred1 != pred2:
                    change_map[i:i+patch_size, j:j+patch_size] = 1

        total_pixels      = change_map.size
        changed_pixels    = np.sum(change_map)
        change_percentage = (changed_pixels / total_pixels) * 100

        return img1, img2, change_map, change_percentage
