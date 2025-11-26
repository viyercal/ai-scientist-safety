import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Setting up working directory
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Synthetic Dataset
class SyntheticEnvDataset(Dataset):
    def __init__(self, size=1000):
        self.data = [(np.random.rand(10), "Sample action") for _ in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action = self.data[idx]
        return torch.FloatTensor(state), action


# Initialize dataset and dataloader
dataset = SyntheticEnvDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load a pretrained LLM
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
language_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Hyperparameters
epochs = 10
learning_rate = 1e-5
optimizer = optim.Adam(language_model.parameters(), lr=learning_rate)

# Experiment data storage
experiment_data = {
    "synthetic_dataset": {
        "metrics": {"train": []},
        "losses": {"train": []},
        "predictions": [],
        "ground_truth": [],
    },
}


# Scenario Diversity Score Calculation
def calculate_sds(predictions):
    unique_predictions = len(set(predictions))
    return unique_predictions / len(predictions)


# Training Loop
for epoch in range(epochs):
    language_model.train()
    total_loss = 0
    predictions = []

    for state, action in dataloader:
        state = state.to(device)
        action_input = tokenizer(
            action, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        outputs = language_model(**action_input, labels=action_input["input_ids"])
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions.append(action)  # Collect predictions

    avg_loss = total_loss / len(dataloader)
    sds = calculate_sds(predictions)
    experiment_data["synthetic_dataset"]["metrics"]["train"].append(
        {"epoch": epoch, "sds": sds}
    )
    experiment_data["synthetic_dataset"]["losses"]["train"].append(avg_loss)

    print(f"Epoch {epoch}: validation_loss = {avg_loss:.4f}, SDS = {sds:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
