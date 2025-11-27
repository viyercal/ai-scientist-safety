import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

for lr, data in experiment_data["learning_rate_tuning"].items():
    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="Training SCS")
        plt.plot(data["metrics"]["val"], label="Validation SCS")
        plt.title(f"SCS Metrics for {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("SCS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"SCS_metrics_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SCS plot for {lr}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.scatter(data["ground_truth"], data["predictions"])
        plt.title(f"Ground Truth vs Predictions for {lr}")
        plt.xlabel("Ground Truth")
        plt.ylabel("Predictions")
        plt.plot(
            [min(data["ground_truth"]), max(data["ground_truth"])],
            [min(data["ground_truth"]), max(data["ground_truth"])],
            color="red",
        )
        plt.savefig(os.path.join(working_dir, f"gt_vs_pred_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating GT vs Predictions plot for {lr}: {e}")
        plt.close()
