import torch
import numpy as np
from PIL import Image
import os

from models.resnet_model import ResNetLandUse
from preprocessing.transforms import get_transforms

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]


class SpatialChangeDetector:

    def __init__(self, model_path, data_dir=None, device=None):

        # Always use CPU on cloud
        self.device = torch.device("cpu")

        # Always use hardcoded classes
        self.classes = EUROSAT_CLASSES

        # Verify model file exists before loading
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                f"Check that download_model.py ran successfully."
            )

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

                if self.predict_patch(patch1) != self.predict_patch(patch2):
                    change_map[i:i+patch_size, j:j+patch_size] = 1

        change_percentage = float(np.sum(change_map) / change_map.size * 100)
        return img1, img2, change_map, change_percentage
