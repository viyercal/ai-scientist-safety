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


# Synthetic Data Generation for varying dimensions
def generate_data(num_samples=1000, dimensions=10):
    states = np.random.rand(num_samples, dimensions)  # varying dimensional state space
    actions = np.random.randint(0, 2, size=num_samples)  # binary actions
    outcomes = states.sum(axis=1) + actions * np.random.rand(
        num_samples
    )  # simplistic outcome
    return states, actions, outcomes


# Model Definition
class SimplePolicyNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SimplePolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
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
    dimensions_list = [5, 10, 15, 20]
    experiment_data = {"multi_dataset_eval": {}}

    learning_rates = [0.001, 0.01, 0.1]
    num_epochs = 50

    for dimensions in dimensions_list:
        print(f"Evaluating for {dimensions} dimensions")
        states, actions, outcomes = generate_data(dimensions=dimensions)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32).to(device)

        train_size = int(0.8 * len(states_tensor))
        val_states_tensor = states_tensor[train_size:]
        val_outcomes_tensor = outcomes_tensor[train_size:]
        states_tensor = states_tensor[:train_size]
        outcomes_tensor = outcomes_tensor[:train_size]

        experiment_data["multi_dataset_eval"][f"dim_{dimensions}"] = {
            "metrics": {"train": [], "val": []},
            "losses": {"train": [], "val": []},
            "predictions": {},
            "ground_truth": {},
        }

        for lr in learning_rates:
            print(f"Training with learning rate: {lr}")
            model = SimplePolicyNetwork(input_dim=dimensions).to(device)
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
                experiment_data["multi_dataset_eval"][f"dim_{dimensions}"]["metrics"][
                    "train"
                ].append(train_scs)
                experiment_data["multi_dataset_eval"][f"dim_{dimensions}"]["losses"][
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
                    experiment_data["multi_dataset_eval"][f"dim_{dimensions}"][
                        "metrics"
                    ]["val"].append(val_scs)
                    experiment_data["multi_dataset_eval"][f"dim_{dimensions}"][
                        "losses"
                    ]["val"].append(val_loss.item())

                print(
                    f"Epoch {epoch + 1}: loss = {loss.item():.4f}, SCS = {train_scs:.4f}, validation_loss = {val_loss.item():.4f}"
                )

            experiment_data["multi_dataset_eval"][f"dim_{dimensions}"]["predictions"][
                f"lr_{lr}"
            ] = val_predictions.cpu().numpy()
            experiment_data["multi_dataset_eval"][f"dim_{dimensions}"]["ground_truth"][
                f"lr_{lr}"
            ] = val_outcomes_tensor.cpu().numpy()

    # Save experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# Run main
main()
