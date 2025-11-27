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

# Plot training and validation losses
for dimension, data in experiment_data["multi_dataset_eval"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"{dimension} - Losses over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dimension}_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dimension}: {e}")
        plt.close()

# Plot validation predictions vs ground truth
learning_rates = ["lr_0.001", "lr_0.01", "lr_0.1"]
for dimension, data in experiment_data["multi_dataset_eval"].items():
    for lr in learning_rates:
        try:
            plt.figure()
            plt.scatter(data["ground_truth"][lr], data["predictions"][lr], alpha=0.5)
            plt.plot(
                [data["ground_truth"][lr].min(), data["ground_truth"][lr].max()],
                [data["ground_truth"][lr].min(), data["ground_truth"][lr].max()],
                "r--",
            )
            plt.title(f"{dimension} - Predictions vs Ground Truth ({lr})")
            plt.xlabel("Ground Truth")
            plt.ylabel("Predictions")
            plt.savefig(
                os.path.join(
                    working_dir, f"{dimension}_predictions_vs_ground_truth_{lr}.png"
                )
            )
            plt.close()
        except Exception as e:
            print(f"Error creating predictions plot for {dimension} at {lr}: {e}")
            plt.close()
