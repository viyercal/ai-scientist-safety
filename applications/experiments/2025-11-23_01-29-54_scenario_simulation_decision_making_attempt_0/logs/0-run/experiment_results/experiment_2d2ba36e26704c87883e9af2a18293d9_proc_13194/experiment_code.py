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
    "optimizer_comparison": {
        "synthetic_data": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

# Parameters for hyperparameter tuning
learning_rate = 0.001
num_epochs = 20

optimizers = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    "Adagrad": optim.Adagrad,
}

for optimizer_name, optimizer_class in optimizers.items():
    print(f"Starting training with optimizer: {optimizer_name}")

    # Initialize model, loss function, optimizer
    model = SimpleLLM().to(device)
    criterion = nn.MSELoss()
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(tensor_data)
        loss = criterion(outputs, tensor_data)  # Predicting future states
        loss.backward()
        optimizer.step()

        experiment_data["optimizer_comparison"]["synthetic_data"]["losses"][
            "train"
        ].append(loss.item())
        if (epoch + 1) % 5 == 0:
            print(
                f"Optimizer {optimizer_name} - Epoch {epoch + 1}: train_loss = {loss.item():.4f}"
            )

        # Calculate a simple metric (SCS)
        scs = 1 - (loss.item() / np.max(data))  # dummy SCS calculation
        experiment_data["optimizer_comparison"]["synthetic_data"]["metrics"][
            "train"
        ].append(scs)

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
