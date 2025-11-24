import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plotting training metrics
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_data"]["metrics"]["train"], label="Training Reward"
    )
    plt.title("Training Metrics for Synthetic Data")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "training_metrics_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training metrics plot: {e}")
    plt.close()

# Plotting validation metrics
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_data"]["metrics"]["val"],
        label="Validation SPE",
        color="orange",
    )
    plt.title("Validation Metrics for Synthetic Data")
    plt.xlabel("Episodes")
    plt.ylabel("SPE (Total Reward - Baseline)")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "validation_metrics_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation metrics plot: {e}")
    plt.close()

# Plotting predicted values over episodes
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_data"]["predictions"], label="Predicted Rewards"
    )
    plt.title("Predicted Rewards over Episodes for Synthetic Data")
    plt.xlabel("Episodes")
    plt.ylabel("Predicted Total Reward")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "predictions_synthetic_data.png"))
    plt.close()
except Exception as e:
    print(f"Error creating predictions plot: {e}")
    plt.close()

# Plotting ground truth if available (Commented out as ground_truth is empty in the current simulation)
# try:
#     plt.figure()
#     plt.plot(experiment_data['synthetic_data']['ground_truth'], label='Ground Truth')
#     plt.title('Ground Truth over Episodes for Synthetic Data')
#     plt.xlabel('Episodes')
#     plt.ylabel('Ground Truth Values')
#     plt.legend()
#     plt.savefig(os.path.join(working_dir, 'ground_truth_synthetic_data.png'))
#     plt.close()
# except Exception as e:
#     print(f"Error creating ground truth plot: {e}")
#     plt.close()
