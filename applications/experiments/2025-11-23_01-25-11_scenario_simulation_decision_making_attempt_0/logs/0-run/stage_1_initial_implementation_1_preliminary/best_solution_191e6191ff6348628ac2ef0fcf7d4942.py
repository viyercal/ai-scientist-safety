import os
import torch
import numpy as np
import random

# Creating a working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Simple synthetic dataset creation
def create_synthetic_data(num_samples=1000, num_features=10):
    data = np.random.rand(num_samples, num_features)
    actions = np.random.randint(0, 5, size=num_samples)  # Random actions between 0-4
    return data, actions


# Function to generate scenarios using a random approach
def generate_scenarios(state, action, num_scenarios=10):
    scenarios = []
    for _ in range(num_scenarios):
        # Simulating different futures based on current state and action
        scenario = state + (action + np.random.normal(0, 0.1, state.shape))  # Add noise
        scenarios.append(scenario)
    return np.array(scenarios)


# Evaluate the Scenario Coverage Ratio
def evaluate_scenario_coverage(scenarios):
    unique_scenarios = len(np.unique(scenarios, axis=0))
    total_possible_scenarios = 10  # Placeholder for maximum possible scenarios
    return unique_scenarios / total_possible_scenarios


# Training loop
def train(num_epochs=10):
    data, actions = create_synthetic_data()
    experiment_data = {
        "synthetic_data": {
            "metrics": {"train": []},
            "scenarios": [],
        }
    }

    for epoch in range(num_epochs):
        epoch_scenarios = []
        for i in range(len(data)):
            state = torch.FloatTensor(data[i]).to(device)
            action = actions[i]
            scenarios = generate_scenarios(state.cpu().numpy(), action)
            epoch_scenarios.append(scenarios)

        # Flatten scenario list for evaluation
        all_scenarios = np.vstack(epoch_scenarios)
        coverage_ratio = evaluate_scenario_coverage(all_scenarios)
        experiment_data["synthetic_data"]["metrics"]["train"].append(coverage_ratio)

        print(f"Epoch {epoch}: Scenario Coverage Ratio = {coverage_ratio:.4f}")

    # Save experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)


# Start training
train(num_epochs=10)
