import torch
import os
from PIL import Image
from preprocessing.transforms import get_transforms
from models.cnn_classifier import LandUseCNN

class ChangeDetector:
    def __init__(self, model_path, data_dir, device):
        self.device = device
        self.data_dir = data_dir
        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        self.model = LandUseCNN(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

        self.transform = get_transforms()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)

        return self.classes[predicted.item()]

    def detect_change(self, image_path_1, image_path_2):
        class1 = self.predict(image_path_1)
        class2 = self.predict(image_path_2)

        changed = class1 != class2

        return class1, class2, changed