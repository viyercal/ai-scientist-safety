# Set random seed
import random
import numpy as np
import torch

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Synthetic dataset generation
np.random.seed(42)
state_space = np.random.rand(1000, 10)  # 1000 samples, 10 features
actions = np.random.randint(0, 2, size=(1000, 1))  # Binary actions
rewards = actions * np.sum(state_space, axis=1, keepdims=True) + np.random.normal(
    0, 0.1, (1000, 1)
)  # Reward function with noise

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    state_space, rewards, test_size=0.2, random_state=42
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Experiment data dictionary
experiment_data = {
    "feature_dimensionality_reduction": {
        "full_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "pca_5d": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "pca_3d": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "pca_2d": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}


# Function to train and evaluate the model
def train_and_evaluate(X_train, X_val, y_train, y_val, experiment_key):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    batch_sizes = [16, 32, 64, 128, 256]
    n_epochs = 50

    for batch_size in batch_sizes:
        print(f"Training {experiment_key} with batch size: {batch_size}")

        model = SimpleNN(X_train.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

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

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                sors = torch.mean(torch.abs(val_outputs - y_val_tensor)).item()

            # Store metrics
            experiment_data["feature_dimensionality_reduction"][experiment_key][
                "losses"
            ]["train"].append(loss.item())
            experiment_data["feature_dimensionality_reduction"][experiment_key][
                "losses"
            ]["val"].append(val_loss.item())
            experiment_data["feature_dimensionality_reduction"][experiment_key][
                "metrics"
            ]["train"].append(sors)
            experiment_data["feature_dimensionality_reduction"][experiment_key][
                "metrics"
            ]["val"].append(sors)

            print(
                f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, SORS: {sors:.4f}"
            )


# Train and evaluate on full dataset
train_and_evaluate(X_train, X_val, y_train, y_val, "full_dataset")

# Apply PCA and reduce dimensionality
pca_list = [5, 3, 2]
for n_components in pca_list:
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    train_and_evaluate(X_train_pca, X_val_pca, y_train, y_val, f"pca_{n_components}d")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
