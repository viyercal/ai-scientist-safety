import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training and validation losses
try:
    plt.figure()
    plt.plot(experiment_data["dynamic_env"]["losses"]["train"], label="Training Loss")
    plt.plot(experiment_data["dynamic_env"]["losses"]["val"], label="Validation Loss")
    plt.title("Loss over Epochs - Dynamic Environment")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dynamic_env_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot predictions vs ground truth
try:
    plt.figure()
    plt.scatter(
        experiment_data["dynamic_env"]["ground_truth"][:50, 0],
        experiment_data["dynamic_env"]["predictions"][:50, 0],
        label="Channel 1",
    )
    plt.scatter(
        experiment_data["dynamic_env"]["ground_truth"][:50, 1],
        experiment_data["dynamic_env"]["predictions"][:50, 1],
        label="Channel 2",
    )
    plt.title("Predictions vs Ground Truth - Dynamic Environment")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dynamic_env_predictions.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()
