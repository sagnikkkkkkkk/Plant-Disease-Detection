import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms
from torchvision import models

from torch.utils.data import (
    DataLoader,
    random_split
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

# =====================================
# Dataset
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    "dataset/PlantVillage",
    transform=transform
)

class_names = dataset.classes

train_size = int(
    0.8 * len(dataset)
)

val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# =====================================
# Model
# =====================================

model = models.mobilenet_v2()

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        model.last_channel,
        len(class_names)
    )
)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

# =====================================
# Prediction
# =====================================

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            predicted.cpu().numpy()
        )

# =====================================
# Accuracy
# =====================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print(
    f"\nOverall Accuracy: {accuracy*100:.2f}%\n"
)

# =====================================
# Classification Report
# =====================================

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# =====================================
# Confusion Matrix
# =====================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(18, 14))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Plant Disease Confusion Matrix"
)

plt.tight_layout()

os.makedirs(
    "outputs",
    exist_ok=True
)

plt.savefig(
    "outputs/confusion_matrix.png"
)

print(
    "\nConfusion Matrix saved to outputs/confusion_matrix.png"
)

# =====================================
# Top 3 Classes
# =====================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

scores = []

for disease in class_names:

    scores.append(
        (
            disease,
            report[disease]["f1-score"]
        )
    )

scores.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 3 Best Performing Diseases:\n")

for disease, score in scores[:3]:

    print(
        f"{disease}: {score*100:.2f}%"
    )

print("\nEvaluation Complete!")