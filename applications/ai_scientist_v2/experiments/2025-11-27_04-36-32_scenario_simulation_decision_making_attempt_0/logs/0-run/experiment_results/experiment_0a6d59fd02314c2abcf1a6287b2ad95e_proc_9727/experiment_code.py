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


# Synthetic datasets with different distributions
class SyntheticDataset1(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)  # 10 features
        self.labels = (self.data.sum(axis=1) > 5).float()  # Different threshold

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SyntheticDataset2(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10) * 2  # Scaled features
        self.labels = (self.data.mean(axis=1) > 0).float()  # Mean threshold

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SyntheticDataset3(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10) + 1  # Shifted features
        self.labels = (self.data.std(axis=1) > 1).float()  # Std threshold

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rates = [0.0001, 0.001, 0.01]

# Prepare experiment data storage
experiment_data = {
    "multiple_synthetic_datasets": {
        "dataset_1": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_2": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
        "dataset_3": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# List of synthetic datasets
datasets = [SyntheticDataset1(), SyntheticDataset2(), SyntheticDataset3()]
dataset_names = ["dataset_1", "dataset_2", "dataset_3"]

# Training loop for each dataset
for ds_idx, dataset in enumerate(datasets):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for lr in learning_rates:
        model = SimpleNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        print(f"Training on {dataset_names[ds_idx]} with learning rate: {lr}")

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(dataloader)
            experiment_data["multiple_synthetic_datasets"][dataset_names[ds_idx]][
                "losses"
            ]["train"].append(avg_train_loss)
            print(f"Epoch {epoch + 1}: training_loss = {avg_train_loss:.4f}")

            # Calculate SES
            predictions = model(dataset.data.to(device)).cpu().detach().numpy()
            ses = np.mean(predictions)
            experiment_data["multiple_synthetic_datasets"][dataset_names[ds_idx]][
                "metrics"
            ]["train"].append(ses)
            experiment_data["multiple_synthetic_datasets"][dataset_names[ds_idx]][
                "predictions"
            ].append(predictions)
            experiment_data["multiple_synthetic_datasets"][dataset_names[ds_idx]][
                "ground_truth"
            ].append(dataset.labels.numpy())
            print(f"Epoch {epoch + 1}: SES = {ses:.4f}")

# Save the experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
