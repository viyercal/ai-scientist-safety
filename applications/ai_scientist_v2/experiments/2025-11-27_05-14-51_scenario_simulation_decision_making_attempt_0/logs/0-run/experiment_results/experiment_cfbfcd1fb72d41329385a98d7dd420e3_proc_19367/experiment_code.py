import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load three HuggingFace datasets
dataset1 = load_dataset("imdb")  # Example dataset 1
dataset2 = load_dataset("ag_news")  # Example dataset 2
dataset3 = load_dataset("squad")  # Example dataset 3

# Example of synthetic environment state/action/reward generation
np.random.seed(42)
state_space = np.random.rand(1000, 10)
actions = np.random.randint(0, 2, size=(1000, 1))
rewards = actions * np.sum(state_space, axis=1, keepdims=True) + np.random.normal(
    0, 0.1, (1000, 1)
)

# Normalize state space
state_space = (state_space - np.mean(state_space, axis=0)) / np.std(state_space, axis=0)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    state_space, rewards, test_size=0.2, random_state=42
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Prepare data tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# Experiment data dictionary
experiment_data = {
    "hyperparam_tuning_batch_size": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Hyperparameter tuning: batch sizes to experiment with
batch_sizes = [16, 32, 64]
n_epochs = 50

for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    total_reward = 0

    for epoch in range(n_epochs):
        model.train()

        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i : i + batch_size]
            y_batch = y_train_tensor[i : i + batch_size]
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            sors = torch.mean(torch.abs(val_outputs - y_val_tensor)).item()

        total_reward += torch.sum(outputs).item()  # Aggregate total rewards for LTRI

        experiment_data["hyperparam_tuning_batch_size"]["synthetic_dataset"]["losses"][
            "train"
        ].append(loss.item())
        experiment_data["hyperparam_tuning_batch_size"]["synthetic_dataset"]["losses"][
            "val"
        ].append(val_loss.item())
        experiment_data["hyperparam_tuning_batch_size"]["synthetic_dataset"]["metrics"][
            "train"
        ].append(sors)
        experiment_data["hyperparam_tuning_batch_size"]["synthetic_dataset"]["metrics"][
            "val"
        ].append(sors)

        print(
            f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, SORS: {sors:.4f}, LTRI: {total_reward / (epoch + 1):.4f}"
        )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
