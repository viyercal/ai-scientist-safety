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

# Plotting training losses
try:
    plt.figure()
    plt.plot(
        experiment_data["simple_scenario"]["losses"]["train"], label="Training Loss"
    )
    plt.title("Training Loss across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "simple_scenario_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plotting training metrics
try:
    plt.figure()
    plt.plot(
        experiment_data["simple_scenario"]["metrics"]["train"],
        label="Training Metrics (SRS)",
    )
    plt.title("Training Metric (SRS) across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "simple_scenario_training_metrics.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training metrics plot: {e}")
    plt.close()
