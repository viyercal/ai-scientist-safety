import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Create synthetic dataset
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = np.random.rand(size, 5).astype(np.float32)  # 5 features
        self.labels = (self.data.sum(axis=1) > 2.5).astype(
            np.float32
        )  # Binary classification
        self.labels = self.labels.reshape(-1, 1)  # Reshape for compatibility

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Create dataset
dataset = SimpleDataset()
train_data, val_data = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Define a simple neural network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Initialize experiment data
experiment_data = {
    "simple_dataset": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training process
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_sas = correct / len(train_loader.dataset)

    # Validation step
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss += criterion(val_outputs, val_labels).item()
            val_predicted = (val_outputs > 0.5).float()
            val_correct += (val_predicted == val_labels).sum().item()

    val_loss /= len(val_loader)
    val_sas = val_correct / len(val_loader.dataset)

    # Record metrics
    experiment_data["simple_dataset"]["metrics"]["train"].append(train_sas)
    experiment_data["simple_dataset"]["metrics"]["val"].append(val_sas)
    experiment_data["losses"]["train"].append(train_loss)
    experiment_data["losses"]["val"].append(val_loss)
    print(
        f"Epoch {epoch + 1}: train_loss = {train_loss:.4f}, train_sas = {train_sas:.4f}, val_loss = {val_loss:.4f}, val_sas = {val_sas:.4f}"
    )

# Save metrics and predictions
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
