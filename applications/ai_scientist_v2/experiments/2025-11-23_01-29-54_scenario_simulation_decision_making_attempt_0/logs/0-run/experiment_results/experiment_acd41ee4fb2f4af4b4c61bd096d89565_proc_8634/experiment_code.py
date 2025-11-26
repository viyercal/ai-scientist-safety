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


# Experiment data storage
experiment_data = {
    "hyperparam_tuning_learning_rate": {
        "synthetic_data": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Parameters for hyperparameter tuning
learning_rates = [0.0001, 0.001, 0.01]
num_epochs = 20

for lr in learning_rates:
    print(f"Starting training with learning rate: {lr}")

    # Initialize model, loss function, optimizer
    model = SimpleLLM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(tensor_data)
        loss = criterion(outputs, tensor_data)  # Predicting future states
        loss.backward()
        optimizer.step()

        experiment_data["hyperparam_tuning_learning_rate"]["synthetic_data"]["losses"][
            "train"
        ].append(loss.item())
        if (epoch + 1) % 5 == 0:
            print(
                f"Learning rate {lr} - Epoch {epoch + 1}: train_loss = {loss.item():.4f}"
            )

        # Calculate a simple metric (SCS)
        scs = 1 - (loss.item() / np.max(data))  # dummy SCS calculation
        experiment_data["hyperparam_tuning_learning_rate"]["synthetic_data"]["metrics"][
            "train"
        ].append(scs)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
