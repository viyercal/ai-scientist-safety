import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleDynamicDataset(Dataset):
    def __init__(self, features=None):
        self.data = np.random.rand(1000, 2).astype(np.float32)
        self.features = features
        if features is not None:
            self.data = self.data[:, features]
        self.labels = (self.data.sum(axis=1) > (len(self.features) / 2)).astype(
            np.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.data[idx]).to(device),
            "label": torch.tensor(self.labels[idx]).to(device),
        }


def add_noise(data, noise_level=0.1):
    noise = np.random.normal(0, noise_level, data.shape).astype(np.float32)
    return data + noise


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


# Experiment Data Structure
experiment_data = {
    "input_feature_modulation": {
        "single_feature": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "both_features": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
        "noise_added": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        },
    }
}

# Different configurations
configurations = {
    "single_feature": {"features": [0], "noise": False},
    "both_features": {"features": [0, 1], "noise": False},
    "noise_added": {"features": [0, 1], "noise": True},
}

# Parameters
num_epochs = 10
batch_sizes = [16, 32]  # Adjust batch sizes for simplicity
loss_function = nn.BCELoss()

for config_name, config in configurations.items():
    print(f"Running configuration: {config_name}")
    dataset = SimpleDynamicDataset(features=config["features"])
    if config["noise"]:
        dataset.data = add_noise(dataset.data)
    model = SimpleNN(input_dim=len(config["features"])).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct_preds = 0

            for batch in train_loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                inputs = batch["features"]
                labels = batch["label"].unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct_preds += (preds.squeeze() == labels.squeeze()).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = correct_preds / len(dataset)

            experiment_data["input_feature_modulation"][config_name]["losses"][
                "train"
            ].append(avg_loss)
            experiment_data["input_feature_modulation"][config_name]["metrics"][
                "train"
            ].append(accuracy)

            print(
                f"{config_name} Epoch {epoch + 1}: train_loss = {avg_loss:.4f}, accuracy = {accuracy:.4f}"
            )

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
