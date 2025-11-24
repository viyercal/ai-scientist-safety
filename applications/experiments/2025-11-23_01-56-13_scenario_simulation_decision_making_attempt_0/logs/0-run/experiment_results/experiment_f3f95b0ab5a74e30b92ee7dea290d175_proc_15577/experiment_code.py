import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SimpleDynamicDataset(Dataset):
    def __init__(self, size=1000):
        self.data = np.random.rand(size, 2).astype(np.float32)
        self.labels = (self.data[:, 0] + self.data[:, 1] > 1).astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.data[idx]),
            "label": torch.tensor(self.labels[idx]),
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


dataset = SimpleDynamicDataset()
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Initialize model, loss function, and optimizer
model = SimpleNN().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

batch_sizes = [16, 32, 64]  # Different batch sizes to test
experiment_data = {
    "batch_size_tuning": {
        "simple_dynamic": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

num_epochs = 10

for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_preds = 0

        for batch in train_loader:
            inputs = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (outputs.squeeze() > 0.5).float()
            correct_preds += (preds == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_preds / len(dataset)

        print(
            f"Epoch {epoch + 1}: train_loss = {avg_loss:.4f}, accuracy = {accuracy:.4f}"
        )

        experiment_data["batch_size_tuning"]["simple_dynamic"]["losses"][
            "train"
        ].append(avg_loss)
        experiment_data["batch_size_tuning"]["simple_dynamic"]["metrics"][
            "train"
        ].append(accuracy)

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
