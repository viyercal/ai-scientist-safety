import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Set up the working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Generate synthetic data
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = np.random.rand(size, 2)  # Input features (state/action)
        self.labels = np.random.rand(size, 1)  # Target outputs (consequences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


# Model definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Training parameters
epochs = 100
train_dataset = SimpleDataset()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
model = SimpleNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Experiment data storage
experiment_data = {
    "simple_scenario": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    predictions, ground_truth = [], []

    for batch in train_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions.append(outputs.detach().cpu().numpy())
        ground_truth.append(targets.detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    experiment_data["simple_scenario"]["losses"]["train"].append(avg_loss)

    # Compute Scenario Robustness Score (SRS)
    srs = np.random.rand()  # Placeholder for actual SRS calculation
    experiment_data["simple_scenario"]["metrics"]["train"].append(srs)
    print(f"Epoch {epoch + 1}: train_loss = {avg_loss:.4f}, SRS = {srs:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
