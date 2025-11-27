import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Prepare working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic Data Generation with noise levels
def generate_data(num_samples=1000, noise_level=0.0):
    states = np.random.rand(num_samples, 10)  # 10-dimensional state space
    actions = np.random.randint(0, 2, size=num_samples)  # binary actions
    outcomes = states.sum(axis=1) + actions * (
        np.random.rand(num_samples) + noise_level
    )  # simplistic outcome
    return states, actions, outcomes


# Model Definition
class SimplePolicyNetwork(nn.Module):
    def __init__(self):
        super(SimplePolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Predicting a single continuous value
        )

    def forward(self, x):
        return self.fc(x)


# SCS Calculation
def calculate_scs(predictions, actuals):
    return np.mean(
        np.abs(predictions - actuals)
    )  # placeholder for inconsistency measure


# Main Function
def main():
    noise_levels = [0.0, 0.5, 1.0]  # Different noise levels for datasets
    experiment_data = {"multi_dataset_evaluation": {}}

    for i, noise in enumerate(noise_levels):
        # Generate data
        states, actions, outcomes = generate_data(noise_level=noise)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32).to(device)

        # Hyperparameter tuning
        learning_rates = [0.001, 0.01, 0.1]
        experiment_data["multi_dataset_evaluation"][f"noise_{noise}"] = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": [],
            "ground_truth": [],
        }

        num_epochs = 50
        train_size = int(0.8 * len(states_tensor))
        val_states_tensor = states_tensor[train_size:]
        val_outcomes_tensor = outcomes_tensor[train_size:]
        states_tensor = states_tensor[:train_size]
        outcomes_tensor = outcomes_tensor[:train_size]

        for lr in learning_rates:
            print(
                f"Training with learning rate: {lr} on dataset with noise level: {noise}"
            )
            model = SimplePolicyNetwork().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                predictions = model(states_tensor).squeeze()
                loss = criterion(predictions, outcomes_tensor)
                loss.backward()
                optimizer.step()

                # Store metrics
                train_scs = calculate_scs(
                    predictions.detach().cpu().numpy(), outcomes_tensor.cpu().numpy()
                )
                experiment_data["multi_dataset_evaluation"][f"noise_{noise}"][
                    "metrics"
                ]["train"].append(train_scs)
                experiment_data["multi_dataset_evaluation"][f"noise_{noise}"]["losses"][
                    "train"
                ].append(loss.item())

                # Validation
                model.eval()
                with torch.no_grad():
                    val_predictions = model(val_states_tensor).squeeze()
                    val_loss = criterion(val_predictions, val_outcomes_tensor)
                    val_scs = calculate_scs(
                        val_predictions.cpu().numpy(), val_outcomes_tensor.cpu().numpy()
                    )
                    experiment_data["multi_dataset_evaluation"][f"noise_{noise}"][
                        "metrics"
                    ]["val"].append(val_scs)
                    experiment_data["multi_dataset_evaluation"][f"noise_{noise}"][
                        "losses"
                    ]["val"].append(val_loss.item())

                print(
                    f"Epoch {epoch + 1}: loss = {loss.item():.4f}, SCS = {train_scs:.4f}, validation_loss = {val_loss.item():.4f}"
                )

            experiment_data["multi_dataset_evaluation"][f"noise_{noise}"][
                "predictions"
            ].append(val_predictions.cpu().numpy())
            experiment_data["multi_dataset_evaluation"][f"noise_{noise}"][
                "ground_truth"
            ].append(val_outcomes_tensor.cpu().numpy())

    # Save experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# Run main
main()
