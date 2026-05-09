import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from preprocessing.dataset import EuroSATDataset
from preprocessing.transforms import get_transforms
from models.resnet_model import ResNetLandUse

train_acc_history = []
val_acc_history   = []
lr_history        = []

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_DIR = "data/raw/2750"

dataset = EuroSATDataset(root_dir=DATA_DIR, transform=get_transforms())

train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

print(f"Total: {len(dataset)}  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = ResNetLandUse(num_classes=len(dataset.classes))
model = model.to(device)

# ── Loss, optimiser, scheduler ────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)   # label smoothing helps generalisation
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# ReduceLROnPlateau: halve LR when val accuracy stops improving for 2 epochs
# This directly fixes the oscillation you saw in the training curve
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",        # we're maximising val_accuracy
    factor=0.5,        # multiply LR by 0.5 on plateau
    patience=2,        # wait 2 epochs before reducing
)

# ── Training ──────────────────────────────────────────────────────────────────
best_val_accuracy = 0.0
epochs = 10  # more epochs are fine now that the scheduler will slow down when needed

os.makedirs("results", exist_ok=True)

for epoch in range(epochs):

    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    train_correct = 0
    train_total   = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents sudden large updates that cause oscillation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        _, predicted  = torch.max(outputs, 1)
        train_total  += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = torch.max(outputs, 1)
            val_total     += labels.size(0)
            val_correct   += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    current_lr   = optimizer.param_groups[0]["lr"]

    print(f"Epoch [{epoch+1}/{epochs}]  "
          f"Train: {train_accuracy:.2f}%  "
          f"Val: {val_accuracy:.2f}%  "
          f"LR: {current_lr:.6f}")

    train_acc_history.append(train_accuracy)
    val_acc_history.append(val_accuracy)
    lr_history.append(current_lr)

    # Step the scheduler using val accuracy (not loss)
    scheduler.step(val_accuracy)

    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "results/best_model.pth")
        print("  ✓ Best model saved!")

print(f"\nTraining complete. Best val accuracy: {best_val_accuracy:.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_acc_history, label="Train Accuracy", color="#4fc3f7", linewidth=2)
ax1.plot(val_acc_history,   label="Val Accuracy",   color="#00e5a0", linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Training vs Validation Accuracy")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(lr_history, color="#ff6b35", linewidth=2, marker="o")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning Rate")
ax2.set_title("Learning Rate Schedule")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/training_curve.png", dpi=150, bbox_inches="tight")
plt.show()