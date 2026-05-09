import torch.nn as nn
import torchvision.models as models


class ResNetLandUse(nn.Module):
    """
    ResNet18 fine-tuned for EuroSAT land-use classification.

    Changes vs original:
    - Dropout(0.3) before the final FC layer  →  reduces overfitting
    - weights= keyword instead of deprecated pretrained=True
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # Use new weights API (suppresses deprecation warning)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.model.fc.in_features

        # Replace classifier head: GlobalAvgPool → Dropout → Linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)