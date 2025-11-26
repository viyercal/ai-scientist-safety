import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic dataset creation
np.random.seed(42)
num_samples = 1000
states = np.random.rand(num_samples, 5)  # 5 features for state
actions = np.random.rand(num_samples, 2)  # 2 possible actions
outcomes = (
    states @ np.array([[0.5], [-0.2], [0.3], [0.1], [0.6]])
    + actions @ np.array([[1.0], [-1.0]])
    + np.random.normal(0, 0.1, (num_samples, 1))
)

# Prepare data loaders
dataset = TensorDataset(
    torch.tensor(states, dtype=torch.float32),
    torch.tensor(actions, dtype=torch.float32),
    torch.tensor(outcomes, dtype=torch.float32),
)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Simple Neural Network Model
class ScenarioModel(nn.Module):
    def __init__(self):
        super(ScenarioModel, self).__init__()
        self.fc1 = nn.Linear(7, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = ScenarioModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Experiment data tracking
experiment_data = {
    "synthetic_data": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}

# Training Loop
for epoch in range(50):
    model.train()
    for batch in train_loader:
        state, action, outcome = [b.to(device) for b in batch]
        inputs = torch.cat((state, action), dim=1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, outcome)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Track loss and accuracy
    experiment_data["synthetic_data"]["losses"]["train"].append(loss.item())
    experiment_data["synthetic_data"]["ground_truth"].extend(
        outcome.detach().cpu().numpy()
    )
    experiment_data["synthetic_data"]["predictions"].extend(
        outputs.detach().cpu().numpy()
    )

    # Print epoch statistics
    print(f"Epoch {epoch + 1}: train_loss = {loss.item():.4f}")

# Compute accuracy (simply comparing means for demonstration, replace with a better metric as per requirements)
predictions = np.array(experiment_data["synthetic_data"]["predictions"])
ground_truth = np.array(experiment_data["synthetic_data"]["ground_truth"])
accuracy = np.mean(np.isclose(predictions, ground_truth, atol=0.1))

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"Scenario Prediction Accuracy: {accuracy:.4f}")
