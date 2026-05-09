import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

from preprocessing.dataset import EuroSATDataset
from preprocessing.transforms import get_transforms
from models.resnet_model import ResNetLandUse

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DATA_DIR = "data/raw/2750"

dataset = EuroSATDataset(DATA_DIR, transform=get_transforms())

loader = DataLoader(dataset, batch_size=32, shuffle=False)

model = ResNetLandUse(num_classes=len(dataset.classes))
model.load_state_dict(torch.load("results/best_model.pth", map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.classes
)

plt.figure(figsize=(10,8))
disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix - Land Use Classification")

plt.show()