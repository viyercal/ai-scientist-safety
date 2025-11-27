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


# Simple synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)  # 10 features
        self.labels = (
            self.data.sum(axis=1) > 0
        ).float()  # Binary classification based on sum

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
learning_rates = [0.0001, 0.001, 0.01]  # Different learning rates for tuning

# Initialize dataset and dataloader
dataset = SyntheticDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Prepare experiment data storage
experiment_data = {
    "hyperparam_tuning_learning_rate": {
        "synthetic_dataset": {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        },
    },
}

# Training loop for different learning rates
for lr in learning_rates:
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"Training with learning rate: {lr}")

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
        experiment_data["hyperparam_tuning_learning_rate"]["synthetic_dataset"][
            "losses"
        ]["train"].append(avg_train_loss)
        print(f"Epoch {epoch + 1}: training_loss = {avg_train_loss:.4f}")

        # Calculate Scenario Evaluation Score (SES)
        predictions = model(dataset.data.to(device)).cpu().detach().numpy()
        ses = np.mean(predictions)  # Simplistic SES for example purpose
        experiment_data["hyperparam_tuning_learning_rate"]["synthetic_dataset"][
            "metrics"
        ]["train"].append(ses)
        print(f"Epoch {epoch + 1}: SES = {ses:.4f}")

# Save the experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
