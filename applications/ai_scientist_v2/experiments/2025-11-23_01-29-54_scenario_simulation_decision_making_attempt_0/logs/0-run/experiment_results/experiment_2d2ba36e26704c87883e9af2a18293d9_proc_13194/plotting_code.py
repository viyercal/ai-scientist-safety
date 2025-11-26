import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    for optimizer_name in experiment_data["optimizer_comparison"]:
        losses = experiment_data["optimizer_comparison"][optimizer_name]["losses"][
            "train"
        ]
        plt.plot(losses, label=optimizer_name)
    plt.title("Training Losses Across Optimizers")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_losses_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training losses plot: {e}")
    plt.close()

try:
    plt.figure()
    for optimizer_name in experiment_data["optimizer_comparison"]:
        metrics = experiment_data["optimizer_comparison"][optimizer_name]["metrics"][
            "train"
        ]
        plt.plot(metrics, label=optimizer_name)
    plt.title("Training Metrics (SCS) Across Optimizers")
    plt.xlabel("Epochs")
    plt.ylabel("SCS Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_metrics_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training metrics plot: {e}")
    plt.close()
