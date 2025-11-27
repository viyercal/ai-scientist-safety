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

for noise in experiment_data["multi_dataset_evaluation"]:
    noise_metrics = experiment_data["multi_dataset_evaluation"][noise]["metrics"]

    try:
        plt.figure()
        plt.plot(noise_metrics["train"], label="Training SCS")
        plt.plot(noise_metrics["val"], label="Validation SCS")
        plt.title(f"SCS Metrics for Noise Level {noise}")
        plt.xlabel("Epochs")
        plt.ylabel("SCS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"scs_metrics_noise_{noise}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SCS metrics plot for noise {noise}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["multi_dataset_evaluation"][noise]["losses"]["train"],
            label="Training Loss",
        )
        plt.plot(
            experiment_data["multi_dataset_evaluation"][noise]["losses"]["val"],
            label="Validation Loss",
        )
        plt.title(f"Loss Metrics for Noise Level {noise}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"loss_metrics_noise_{noise}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss metrics plot for noise {noise}: {e}")
        plt.close()
