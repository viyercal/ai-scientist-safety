import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


# Function to generate synthetic datasets
def generate_synthetic_dataset(seed, num_samples=1000, num_features=10):
    np.random.seed(seed)
    state_space = np.random.rand(num_samples, num_features)
    actions = np.random.randint(0, 2, size=(num_samples, 1))
    rewards = actions * np.sum(state_space, axis=1, keepdims=True) + np.random.normal(
        0, 0.1, (num_samples, 1)
    )
    return state_space, rewards


# Generate three distinct synthetic datasets
datasets = {
    "dataset_1": generate_synthetic_dataset(42),
    "dataset_2": generate_synthetic_dataset(43),
    "dataset_3": generate_synthetic_dataset(44),
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Experiment data dictionary
experiment_data = {"multi_dataset_evaluation": {}}

# Hyperparameter tuning: batch sizes to experiment with
batch_sizes = [16, 32, 64, 128, 256]
n_epochs = 50

# Train and evaluate on each dataset
for dataset_name, (state_space, rewards) in datasets.items():
    print(f"Evaluating {dataset_name}")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        state_space, rewards, test_size=0.2, random_state=42
    )

    # Prepare data tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    experiment_data["multi_dataset_evaluation"][dataset_name] = {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")

        # Initialize model and optimizer for each batch size
        model = SimpleNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        for epoch in range(n_epochs):
            model.train()

            # Mini-batch training
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

                # Calculate SORS
                sors = torch.mean(torch.abs(val_outputs - y_val_tensor)).item()

            # Store metrics
            experiment_data["multi_dataset_evaluation"][dataset_name]["losses"][
                "train"
            ].append(loss.item())
            experiment_data["multi_dataset_evaluation"][dataset_name]["losses"][
                "val"
            ].append(val_loss.item())
            experiment_data["multi_dataset_evaluation"][dataset_name]["metrics"][
                "train"
            ].append(sors)
            experiment_data["multi_dataset_evaluation"][dataset_name]["metrics"][
                "val"
            ].append(sors)

            print(
                f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, SORS: {sors:.4f}"
            )

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
