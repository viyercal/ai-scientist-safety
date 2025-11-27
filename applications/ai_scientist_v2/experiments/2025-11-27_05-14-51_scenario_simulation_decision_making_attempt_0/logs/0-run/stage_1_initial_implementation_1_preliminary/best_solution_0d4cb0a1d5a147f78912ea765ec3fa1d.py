import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create synthetic dataset
num_samples = 1000
state = np.random.rand(num_samples, 4)  # Random states
action = np.random.randint(0, 2, size=(num_samples, 1))  # Random binary actions
reward = np.random.rand(num_samples)  # Random rewards
data = np.hstack([state, action, reward.reshape(-1, 1)])


# Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.fc(x)


model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Initialize metrics
experiment_data = {
    "dataset_name_1": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
}

num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    # Prepare training data
    indices = np.random.choice(num_samples, 32)
    batch_data = torch.tensor(data[indices], dtype=torch.float32).to(device)

    # Forward pass
    inputs = batch_data[:, :-1]
    targets = batch_data[:, -1]
    outputs = model(inputs)

    # Compute loss
    loss = loss_fn(outputs, targets.view(-1, 1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Store metrics
    experiment_data["dataset_name_1"]["losses"]["train"].append(loss.item())

    # Validation phase (basic validation)
    model.eval()
    val_indices = np.random.choice(num_samples, 32)
    val_data = torch.tensor(data[val_indices], dtype=torch.float32).to(device)
    val_inputs = val_data[:, :-1]
    val_targets = val_data[:, -1]
    val_outputs = model(val_inputs)
    val_loss = loss_fn(val_outputs, val_targets.view(-1, 1))

    # Update metrics
    experiment_data["dataset_name_1"]["losses"]["val"].append(val_loss.item())
    print(f"Epoch {epoch}: validation_loss = {val_loss:.4f}")

# Calculate SORS (Simple average for demonstration; replace with a proper calculation later)
sors = np.mean(experiment_data["dataset_name_1"]["losses"]["val"])
experiment_data["dataset_name_1"]["metrics"]["train"].append(sors)

# Save data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
