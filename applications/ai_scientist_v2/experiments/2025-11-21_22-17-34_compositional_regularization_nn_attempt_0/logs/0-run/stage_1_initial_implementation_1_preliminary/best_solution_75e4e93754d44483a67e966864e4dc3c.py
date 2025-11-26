import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.01
CREG_ALPHA = 0.1  # compositional regularization weight


# Create a synthetic dataset for compositional generalization
class SyntheticDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = np.random.randint(0, 10, (size, 2))  # Two features
        self.labels = (self.data[:, 0] + self.data[:, 1]) % 10  # Sum modulo 10

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


train_dataset = SyntheticDataset(2000)
val_dataset = SyntheticDataset(500)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()


# Define a custom loss function incorporating compositional regularization
def compositional_loss(outputs, labels, model):
    cross_entropy_loss = criterion(outputs, labels)
    # Here we can add a simple regularization term based on model weights
    regularization_term = CREG_ALPHA * torch.sum(torch.pow(model.fc1.weight, 2))
    return cross_entropy_loss + regularization_term


# Training loop
metrics = {"epoch": [], "val_accuracy": []}
for epoch in range(NUM_EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = compositional_loss(outputs, labels, model)
        loss.backward()
        optimizer.step()

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    metrics["epoch"].append(epoch + 1)
    metrics["val_accuracy"].append(val_accuracy)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Validation Accuracy: {val_accuracy:.2f}%")

# Save metrics
np.save(os.path.join(working_dir, "experiment_data.npy"), metrics)
