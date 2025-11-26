import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

activation_functions = ["ReLU", "Sigmoid", "Tanh"]

# Plot training losses
for function in activation_functions:
    try:
        losses = experiment_data["activation_function_comparison"]["synthetic_data"][
            "losses"
        ]["train"]
        plt.figure()
        plt.plot(losses, label=f"{function} Loss")
        plt.title(f"Training Loss for {function}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_{function}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {function}: {e}")
        plt.close()

# Plot training metrics
for function in activation_functions:
    try:
        metrics = experiment_data["activation_function_comparison"]["synthetic_data"][
            "metrics"
        ]["train"]
        plt.figure()
        plt.plot(metrics, label=f"{function} Metric")
        plt.title(f"Training Metric for {function}")
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_metric_{function}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {function}: {e}")
        plt.close()
