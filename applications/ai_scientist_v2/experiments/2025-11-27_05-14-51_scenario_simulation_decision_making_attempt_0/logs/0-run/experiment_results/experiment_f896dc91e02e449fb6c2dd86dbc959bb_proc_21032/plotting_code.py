import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses for each dataset
for key in experiment_data["feature_dimensionality_reduction"]:
    try:
        losses_train = experiment_data["feature_dimensionality_reduction"][key][
            "losses"
        ]["train"]
        losses_val = experiment_data["feature_dimensionality_reduction"][key]["losses"][
            "val"
        ]
        plt.figure()
        plt.plot(losses_train, label="Training Loss")
        plt.plot(losses_val, label="Validation Loss")
        plt.title(f"{key} Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{key}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {key}: {e}")
        plt.close()

# Plot metrics for training and validation for each dataset
for key in experiment_data["feature_dimensionality_reduction"]:
    try:
        metrics_train = experiment_data["feature_dimensionality_reduction"][key][
            "metrics"
        ]["train"]
        metrics_val = experiment_data["feature_dimensionality_reduction"][key][
            "metrics"
        ]["val"]
        plt.figure()
        plt.plot(metrics_train, label="Training Metric (SORS)")
        plt.plot(metrics_val, label="Validation Metric (SORS)")
        plt.title(f"{key} Metric Curves")
        plt.xlabel("Epoch")
        plt.ylabel("SORS Metric")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{key}_metric_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating metric plot for {key}: {e}")
        plt.close()
