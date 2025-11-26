import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Training Loss Curve
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_data"]["losses"]["train"], label="Training Loss"
    )
    plt.title("Training Loss Curve")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss curve: {e}")
    plt.close()

# Predictions vs Ground Truth
try:
    predictions = np.array(experiment_data["synthetic_data"]["predictions"])
    ground_truth = np.array(experiment_data["synthetic_data"]["ground_truth"])

    plt.figure()
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.title("Predictions vs Ground Truth")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.plot(
        [ground_truth.min(), ground_truth.max()],
        [ground_truth.min(), ground_truth.max()],
        "r--",
    )  # 45-degree line
    plt.savefig(
        os.path.join(working_dir, "synthetic_data_predictions_vs_ground_truth.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating predictions vs ground truth plot: {e}")
    plt.close()
