import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_dataset"]["losses"]["train"], label="Training Loss"
    )
    plt.plot(
        experiment_data["synthetic_dataset"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_dataset"]["metrics"]["train"],
        label="Training Metrics",
    )
    plt.plot(
        experiment_data["synthetic_dataset"]["metrics"]["val"],
        label="Validation Metrics",
    )
    plt.title("Training and Validation Metrics (SORS)")
    plt.xlabel("Epochs")
    plt.ylabel("SORS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_metrics_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metrics plot: {e}")
    plt.close()
