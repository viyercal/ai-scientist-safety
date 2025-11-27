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

for activation_name, activation_data in experiment_data[
    "activation_function_variation"
].items():
    for lr_name, metrics_data in activation_data.items():
        train_loss = metrics_data["losses"]["train"]
        val_loss = metrics_data["losses"]["val"]
        train_metric = metrics_data["metrics"]["train"]
        val_metric = metrics_data["metrics"]["val"]

        try:
            # Training and Validation Loss Plot
            plt.figure()
            plt.plot(train_loss, label="Training Loss")
            plt.plot(val_loss, label="Validation Loss")
            plt.title(f"{activation_name} - Learning Rate {lr_name} - Loss Curves")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"{working_dir}/{activation_name}_{lr_name}_loss_curves.png")
            plt.close()
        except Exception as e:
            print(f"Error creating Loss plot: {e}")
            plt.close()

        try:
            # Training and Validation Metrics Plot
            plt.figure()
            plt.plot(train_metric, label="Training Metric")
            plt.plot(val_metric, label="Validation Metric")
            plt.title(f"{activation_name} - Learning Rate {lr_name} - Metric Curves")
            plt.xlabel("Epochs")
            plt.ylabel("Metric (SCS)")
            plt.legend()
            plt.savefig(f"{working_dir}/{activation_name}_{lr_name}_metric_curves.png")
            plt.close()
        except Exception as e:
            print(f"Error creating Metric plot: {e}")
            plt.close()
