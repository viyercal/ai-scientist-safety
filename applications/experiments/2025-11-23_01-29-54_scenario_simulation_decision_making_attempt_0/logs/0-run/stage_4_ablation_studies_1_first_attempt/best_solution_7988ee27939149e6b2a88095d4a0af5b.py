import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_synthetic_data(num_samples=1000, seq_length=10):
    return np.random.rand(num_samples, seq_length)


data = generate_synthetic_data()
tensor_data = torch.tensor(data, dtype=torch.float32).to(device)


class SimpleLLM(nn.Module):
    def __init__(self, activation_function):
        super(SimpleLLM, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 10)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        return self.fc2(x)


experiment_data = {
    "activation_function_comparison": {
        "synthetic_data": {
            "metrics": {"train": []},
            "losses": {"train": []},
            "predictions": [],
            "ground_truth": [],
        }
    }
}

learning_rates = [0.0001, 0.001, 0.01]
num_epochs = 20
activation_functions = {
    "ReLU": torch.relu,
    "Sigmoid": torch.sigmoid,
    "Tanh": torch.tanh,
}

for name, activation in activation_functions.items():
    for lr in learning_rates:
        print(
            f"Starting training with activation function: {name}, learning rate: {lr}"
        )

        model = SimpleLLM(activation).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(tensor_data)
            loss = criterion(outputs, tensor_data)
            loss.backward()
            optimizer.step()

            experiment_data["activation_function_comparison"]["synthetic_data"][
                "losses"
            ]["train"].append(loss.item())
            if (epoch + 1) % 5 == 0:
                print(
                    f"{name} - Learning rate {lr} - Epoch {epoch + 1}: train_loss = {loss.item():.4f}"
                )

            scs = 1 - (loss.item() / np.max(data))
            experiment_data["activation_function_comparison"]["synthetic_data"][
                "metrics"
            ]["train"].append(scs)

np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
