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
import random

# Setting up the working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Synthetic dataset creation
num_states = 100
num_actions = 5
np.random.seed(42)
states = np.random.rand(num_states, 5)  # 5 features for each state
actions = np.random.randint(0, num_actions, size=num_states)  # Random actions


# A function to simulate rewards
def get_reward(state, action):
    return np.dot(state, np.random.rand(5)) + (action * 0.1)  # Simple reward mechanism


# Training and evaluation setup
num_episodes = 100
metrics = {"train": [], "val": []}
experiment_data = {
    "synthetic_data": {"metrics": metrics, "predictions": [], "ground_truth": []}
}

# Basic reinforcement learning with scenario planning
for episode in range(num_episodes):
    total_reward = 0
    for state in states:
        action = random.choice(actions)  # Agent selects action randomly
        reward = get_reward(state, action)
        total_reward += reward

        # Here we simulate a scenario with an LLM (placeholder)
        # The actual LLM would generate alternatives (we skip LLM for simplicity)

    # Store the total reward for the episode
    experiment_data["synthetic_data"]["predictions"].append(total_reward)
    metrics["train"].append(total_reward)

    # Calculate SPE as a comparison against baseline (average reward)
    baseline = np.mean([get_reward(state, random.choice(actions)) for state in states])
    SPE = total_reward - baseline  # naive SPE calculation
    metrics["val"].append(SPE)

    print(f"Episode {episode+1}: total_reward = {total_reward:.4f}, SPE = {SPE:.4f}")

# Save results
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
