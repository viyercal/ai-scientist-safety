import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Setting up the working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate synthetic dataset
num_samples = 1000
states = np.random.rand(num_samples, 5)  # 5 features representing states
actions = np.random.randint(0, 3, num_samples)  # 3 possible actions
rewards = np.random.rand(num_samples)  # random rewards for demonstration

# Create DataLoader
dataset = TensorDataset(
    torch.FloatTensor(states), torch.LongTensor(actions), torch.FloatTensor(rewards)
)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Simple neural network model for Q-learning
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 3)  # 3 actions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training configuration
num_epochs = 50
experiment_data = {
    "synthetic_data": {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    },
}


def calculate_scenario_decision_quality(predictions, ground_truth):
    # Dummy SDQ metric calculation (higher is better)
    return np.mean(predictions == ground_truth)


# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for state, action, reward in data_loader:
        state, action, reward = state.to(device), action.to(device), reward.to(device)

        optimizer.zero_grad()
        q_values = model(state)
        loss = criterion(q_values.gather(1, action.unsqueeze(-1)), reward.unsqueeze(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    experiment_data["synthetic_data"]["losses"]["train"].append(avg_loss)

    # Simulated predictions (randomly selected predictions for the sake of example)
    with torch.no_grad():
        model.eval()
        preds = model(torch.FloatTensor(states).to(device)).cpu().numpy()
        predicted_actions = np.argmax(preds, axis=1)
        sdq = calculate_scenario_decision_quality(predicted_actions, actions)
        experiment_data["synthetic_data"]["metrics"]["train"].append(sdq)

    print(f"Epoch {epoch}: training_loss = {avg_loss:.4f}, SDQ = {sdq:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
