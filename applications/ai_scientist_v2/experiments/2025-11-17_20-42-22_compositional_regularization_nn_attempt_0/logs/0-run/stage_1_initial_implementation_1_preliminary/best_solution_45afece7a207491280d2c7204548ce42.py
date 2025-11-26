import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18
from datasets import load_dataset

# ====================================================
# Setup GPU/CPU
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================================================
# Config
# ====================================================
DATASET_HF_NAME = "timm/mini-imagenet"
IMAGE_SIZE = 84

BATCH_SIZE = 64
NUM_EPOCHS = 1

LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-2
MOMENTUM = 0.9

MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 500
MAX_TEST_SAMPLES = 500

# Create working directory to save metrics
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# ====================================================
# Data
# ====================================================
transform = T.Compose(
    [
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_split(split_name, max_samples=None):
    ds = load_dataset(DATASET_HF_NAME, split=split_name)
    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


train_hf = load_split("train", MAX_TRAIN_SAMPLES)
val_hf = load_split("validation", MAX_VAL_SAMPLES)
test_hf = load_split("test", MAX_TEST_SAMPLES)


class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        img = sample["image"]
        label = int(sample["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


train_dataset = HFImageDataset(train_hf, transform)
val_dataset = HFImageDataset(val_hf, transform)
test_dataset = HFImageDataset(test_hf, transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# ====================================================
# Model
# ====================================================
model = resnet18(weights=None)
num_features = model.fc.in_features
num_classes = int(max(train_hf["label"])) + 1
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)

# ====================================================
# Training loop
# ====================================================
metrics = {
    "epoch": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "test_accuracy": [],
}

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for step, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    val_accuracy = 100.0 * (outputs.argmax(dim=1) == labels).sum().item() / len(labels)
    metrics["epoch"].append(epoch)
    metrics["val_accuracy"].append(val_accuracy)

    print(f"Epoch {epoch + 1}: validation_accuracy = {val_accuracy:.4f}")

# Save metrics data
np.save(os.path.join(working_dir, "experiment_data.npy"), metrics)
