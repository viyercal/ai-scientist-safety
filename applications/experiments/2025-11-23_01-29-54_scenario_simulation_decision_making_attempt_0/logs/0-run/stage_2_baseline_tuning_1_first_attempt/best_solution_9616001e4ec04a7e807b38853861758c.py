import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic data generation
def generate_synthetic_data(num_samples=1000, seq_length=10):
    return np.random.rand(num_samples, seq_length)


data = generate_synthetic_data()
tensor_data = torch.tensor(data, dtype=torch.float32).to(device)


# Simple neural network model
class SimpleLLM(nn.Module):
    def __init__(self):
        super(SimpleLLM, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Initialize model, loss function, optimizer
criterion = nn.MSELoss()
batch_sizes = [1, 16, 32, 64]  # Different batch sizes to tune
experiment_data = {"batch_size_tuning": {}}

num_epochs = 20
for batch_size in batch_sizes:
    dataloader = DataLoader(
        TensorDataset(tensor_data), batch_size=batch_size, shuffle=True
    )
    model = SimpleLLM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"] = {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    }

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Predicting future states
            loss.backward()
            optimizer.step()

            experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["losses"][
                "train"
            ].append(loss.item())
            if (epoch + 1) % 5 == 0:
                print(
                    f"Batch Size: {batch_size}, Epoch {epoch + 1}: train_loss = {loss.item():.4f}"
                )

            # SCS computation (simplified)
            scs = 1 - (loss.item() / np.max(data))  # dummy SCS calculation
            experiment_data["batch_size_tuning"][f"batch_size_{batch_size}"]["metrics"][
                "train"
            ].append(scs)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
