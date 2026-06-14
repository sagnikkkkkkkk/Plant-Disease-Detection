import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# =========================
# Create folders
# =========================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================
# Configuration
# =========================

DATASET_PATH = "dataset/PlantVillage"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# =========================
# GPU Detection
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# =========================
# Image Transformations
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Dataset
# =========================

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

class_names = dataset.classes

np.save(
    "models/class_names.npy",
    class_names
)

print("\nClasses:")
for c in class_names:
    print(c)

# =========================
# Train / Validation Split
# =========================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\nTraining Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))

# =========================
# MobileNetV2 Transfer Learning
# =========================

model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.DEFAULT
)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        model.last_channel,
        len(class_names)
    )
)

model = model.to(device)

# =========================
# Loss & Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)

# =========================
# Training Loop
# =========================

best_val_acc = 0

train_acc_history = []
val_acc_history = []

train_loss_history = []
val_loss_history = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(
            labels
        ).sum().item()

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # Validation

    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(
                labels
            ).sum().item()

    val_acc = 100 * correct / total
    val_loss /= len(val_loader)

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "models/best_model.pth"
        )

        print(
            "Model Saved!"
        )

# =========================
# Accuracy Plot
# =========================

plt.figure(figsize=(10,5))

plt.plot(train_acc_history)
plt.plot(val_acc_history)

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Train",
    "Validation"
])

plt.savefig(
    "outputs/accuracy.png"
)

# =========================
# Loss Plot
# =========================

plt.figure(figsize=(10,5))

plt.plot(train_loss_history)
plt.plot(val_loss_history)

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Train",
    "Validation"
])

plt.savefig(
    "outputs/loss.png"
)

print("\nTraining Complete!")
print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.2f}%"
)