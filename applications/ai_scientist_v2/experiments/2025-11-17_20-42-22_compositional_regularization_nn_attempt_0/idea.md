## Name

compositional_regularization_nn

## Title

Enhancing Compositional Generalization in Neural Networks via Compositional Regularization

## Short Hypothesis

Introducing a compositional regularization term during training can encourage neural networks to develop compositional representations, thereby improving their ability to generalize to novel combinations of known components.

## Related Work

Previous work has highlighted the challenges neural networks face in achieving compositional generalization. Studies such as 'Compositional Generalization through Abstract Representations in Human and Artificial Neural Networks' (Ito et al., NeurIPS 2022) have explored abstract representations to tackle this issue. However, limited research focuses on directly incorporating explicit regularization terms into the training objective to enforce compositional structures. Our proposal distinguishes itself by introducing a novel regularization approach that penalizes deviations from predefined compositional patterns during training, encouraging the network to internalize compositional rules.

## Abstract

Neural networks excel in many tasks but often struggle with compositional generalization—the ability to understand and generate novel combinations of familiar components. This limitation hampers their performance on tasks requiring systematic generalization beyond the training data. In this proposal, we introduce a novel training method that incorporates an explicit compositional regularization term into the loss function of neural networks. This regularization term is designed to encourage the formation of compositional representations by penalizing the network when its internal representations deviate from expected compositional structures. We hypothesize that this approach will enhance the network's ability to generalize to unseen combinations, mimicking human-like compositional reasoning. We will test our method on synthetic benchmarks like the SCAN and COGS datasets, which are specifically designed to evaluate compositional generalization, as well as on real-world tasks such as machine translation and semantic parsing. By comparing our method to baseline models and existing approaches, we aim to demonstrate significant improvements in generalization performance. This work offers a new avenue for enforcing compositionality in neural networks through regularization, potentially bridging the gap between neural network capabilities and human cognitive flexibility.

## Experiments

- Implement the compositional regularization term and integrate it into the loss function of standard sequence-to-sequence neural network architectures with attention mechanisms.
- Train models on synthetic datasets like SCAN and COGS, evaluating performance on compositional generalization tasks with and without the regularization term.
- Apply the method to real-world tasks such as machine translation using the IWSLT dataset and semantic parsing with the GeoQuery dataset, assessing improvements in generalization to new language constructs.
- Analyze the learned representations by visualizing embedding spaces and utilizing compositionality metrics to assess how the regularization affects internal representations.
- Conduct ablation studies to determine the impact of different strengths of the regularization term, identifying the optimal balance between enforcing compositionality and maintaining overall performance.
- Compare the proposed method against other approaches aimed at improving compositional generalization, such as meta-learning techniques and specialized architectures.

## Risk Factors And Limitations

- The effectiveness of the compositional regularization may vary across different datasets and tasks, potentially limiting its generalizability.
- An improperly balanced regularization term could negatively impact model performance on the primary task, leading to lower accuracy.
- Additional computational overhead from calculating the regularization term may increase training time and resource requirements.
- Defining appropriate compositional structures for complex or less-understood domains may be challenging, affecting the applicability of the method.
- The approach may face scalability issues when applied to very large models or datasets common in industrial applications.

## Code To Potentially Use

Use the following code as context for your experiments:

```python
import time
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18  # lighter than resnet50

from datasets import load_dataset

# ====================================================
# Config
# ====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For the initial implementation, keep it simple and small.
DATASET_HF_NAME = "timm/mini-imagenet"
IMAGE_SIZE = 84

BATCH_SIZE = 64
NUM_EPOCHS = 1

LEARNING_RATE = 0.1
WEIGHT_DECAY = 1e-2
MOMENTUM = 0.9

STEPS_TO_LOG = 20

# Limit dataset size so the run finishes comfortably in the sandbox
MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 500
MAX_TEST_SAMPLES = 500

# Logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"metrics_mini_imagenet_resnet18_{timestamp}.npy"

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
        # IMPORTANT: index into the HF dataset
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
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# ====================================================
# Model
# ====================================================
model = resnet18(weights=None)

# Set correct number of classes from labels
num_features = model.fc.in_features
num_classes = int(max(train_hf["label"])) + 1
model.fc = nn.Linear(num_features, num_classes)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)

# Simple step LR schedule
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# ====================================================
# Accuracy helper (no model.eval() to avoid sandbox eval() weirdness)
# ====================================================
def evaluate_accuracy(model, data_loader, device, max_batches=None):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            if (max_batches is not None) and (batch_idx >= max_batches):
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    if total == 0:
        return 0.0
    return 100.0 * correct / total


# ====================================================
# Training loop
# ====================================================
start_time = time.time()

metrics = {
    "epoch": [],
    "step": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "test_accuracy": [],
}

global_step = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    epoch_start = time.time()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step += 1

        if (step + 1) % STEPS_TO_LOG == 0:
            avg_loss = running_loss / STEPS_TO_LOG
            running_loss = 0.0

            # Quick eval on a few batches for speed
            train_acc = evaluate_accuracy(model, train_loader, DEVICE, max_batches=5)
            val_acc = evaluate_accuracy(model, val_loader, DEVICE, max_batches=5)
            test_acc = evaluate_accuracy(model, test_loader, DEVICE, max_batches=5)

            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                f"Step [{step+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Train Acc: {train_acc:.2f}% "
                f"Val Acc: {val_acc:.2f}% "
                f"Test Acc: {test_acc:.2f}% "
                f"Elapsed: {elapsed:.1f}s"
            )

            metrics["epoch"].append(epoch + 1)
            metrics["step"].append(global_step)
            metrics["train_loss"].append(avg_loss)
            metrics["train_accuracy"].append(train_acc)
            metrics["val_accuracy"].append(val_acc)
            metrics["test_accuracy"].append(test_acc)

            # Save metrics for AI-Scientist
            np.save(LOG_FILE, metrics)

    scheduler.step()

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} finished in {epoch_time:.1f}s")

total_time = time.time() - start_time
print(f"Training completed in {total_time:.1f}s")

# Final full evaluation
final_train_acc = evaluate_accuracy(model, train_loader, DEVICE, max_batches=None)
final_val_acc = evaluate_accuracy(model, val_loader, DEVICE, max_batches=None)
final_test_acc = evaluate_accuracy(model, test_loader, DEVICE, max_batches=None)

print(
    f"Final Accuracies -> Train: {final_train_acc:.2f}%, "
    f"Val: {final_val_acc:.2f}%, Test: {final_test_acc:.2f}%"
)

metrics["final_train_accuracy"] = final_train_acc
metrics["final_val_accuracy"] = final_val_acc
metrics["final_test_accuracy"] = final_test_acc

np.save(LOG_FILE, metrics)

```

