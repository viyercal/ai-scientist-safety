import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Prepare device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic dataset
class DynamicEnvDataset(Dataset):
    def __init__(self, size=1000, augment=False):
        self.x = np.random.rand(size, 2)  # Current state with 2 features
        self.y = self.x * 2 + np.random.normal(0, 0.1, self.x.shape)  # Future state
        if augment:
            self.x += np.random.normal(
                0, 0.05, self.x.shape
            )  # Add noise for augmentation

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(
            self.y[idx], dtype=torch.float32
        )


# Function to train the model and record metrics
def train_model(train_loader, val_loader, epochs):
    model = ScenarioModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    experiment_results = {
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }

    # Training and validation loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        experiment_results["losses"]["train"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                experiment_results["predictions"].append(outputs.cpu().numpy())
                experiment_results["ground_truth"].append(targets.cpu().numpy())

        val_loss /= len(val_loader)
        experiment_results["losses"]["val"].append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: validation_loss = {val_loss:.4f}")

    return experiment_results


# Define simple model
class ScenarioModel(nn.Module):
    def __init__(self):
        super(ScenarioModel, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Create datasets and dataloaders for both augmented and non-augmented
datasets = {
    "Without Augmentation": DynamicEnvDataset(),
    "With Augmentation": DynamicEnvDataset(augment=True),
}

experiment_data = {"data_augmentation_impact": {}}

# Loop through datasets
for name, dataset in datasets.items():
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    experiment_data["data_augmentation_impact"][name] = train_model(
        train_loader, val_loader, epochs=10
    )

# Saving experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
