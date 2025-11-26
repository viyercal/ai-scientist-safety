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
    for config_name in experiment_data["input_feature_modulation"]:
        metrics = experiment_data["input_feature_modulation"][config_name]["metrics"][
            "train"
        ]
        losses = experiment_data["input_feature_modulation"][config_name]["losses"][
            "train"
        ]

        plt.figure()
        plt.plot(range(1, len(metrics) + 1), metrics, label="Accuracy")
        plt.title(f"{config_name} - Training Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{config_name}_training_metrics.png"))
        plt.close()

        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, label="Loss", color="orange")
        plt.title(f"{config_name} - Training Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{config_name}_training_losses.png"))
        plt.close()

        if (
            len(experiment_data["input_feature_modulation"][config_name]["predictions"])
            > 0
        ):
            predictions = experiment_data["input_feature_modulation"][config_name][
                "predictions"
            ][:5]
            ground_truth = experiment_data["input_feature_modulation"][config_name][
                "ground_truth"
            ][:5]
            plt.figure()
            plt.plot(predictions, label="Predictions")
            plt.plot(ground_truth, label="Ground Truth", linestyle="dashed")
            plt.title(f"{config_name} - Predictions vs Ground Truth")
            plt.xlabel("Samples")
            plt.ylabel("Values")
            plt.legend()
            plt.savefig(
                os.path.join(working_dir, f"{config_name}_predictions_vs_truth.png")
            )
            plt.close()
except Exception as e:
    print(f"Error creating plots: {e}")
