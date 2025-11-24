import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Create working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic Dataset
class DynamicEnvDataset(Dataset):
    def __init__(self, size):
        self.states = np.random.rand(size, 10)  # 10-dimensional states
        self.actions = np.random.randint(0, 2, size)  # Binary actions
        self.rewards = np.random.rand(size)  # Random rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return {
            "state": torch.tensor(self.states[index], dtype=torch.float32).to(device),
            "action": torch.tensor(self.actions[index], dtype=torch.float32).to(device),
            "reward": torch.tensor(self.rewards[index], dtype=torch.float32).to(device),
        }


# Simple Neural Network for Decision Making
class SimpleDecider(nn.Module):
    def __init__(self):
        super(SimpleDecider, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)  # Two actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Training parameters
num_epochs = 5
batch_size = 32
dataset_size = 1000

# Initialize dataset and dataloader
dataset = DynamicEnvDataset(dataset_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, optimizer
model = SimpleDecider().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data storage
experiment_data = {
    "dynamic_env": {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }
}

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_sdq = 0

    for batch in dataloader:
        optimizer.zero_grad()

        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]

        # Forward pass
        outputs = model(states)

        # Calculate loss (mean squared error)
        loss = criterion(outputs, actions.unsqueeze(1))
        total_loss += loss.item()

        # Simulated SDQ calculation (simplistic approach)
        sdq = torch.mean((outputs.argmax(dim=1) == actions).float())
        total_sdq += sdq.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    avg_sdq = total_sdq / len(dataloader)

    experiment_data["dynamic_env"]["losses"]["train"].append(avg_loss)
    experiment_data["dynamic_env"]["metrics"]["train"].append(avg_sdq)

    print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}, SDQ = {avg_sdq:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
