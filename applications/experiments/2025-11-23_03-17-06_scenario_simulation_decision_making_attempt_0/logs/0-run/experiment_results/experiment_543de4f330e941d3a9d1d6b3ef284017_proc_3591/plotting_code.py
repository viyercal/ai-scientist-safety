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

for dataset_name, results in experiment_data["data_augmentation_impact"].items():
    try:
        plt.figure()
        plt.plot(results["losses"]["train"], label="Training Loss")
        plt.plot(results["losses"]["val"], label="Validation Loss")
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss curves for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        predictions = np.concatenate(results["predictions"])
        ground_truth = np.concatenate(results["ground_truth"])
        plt.scatter(
            ground_truth[:, 0], ground_truth[:, 1], label="Ground Truth", alpha=0.5
        )
        plt.scatter(
            predictions[:, 0], predictions[:, 1], label="Generated Samples", alpha=0.5
        )
        plt.title(f"{dataset_name} - Ground Truth vs Generated Samples")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_gt_vs_generated.png"))
        plt.close()
    except Exception as e:
        print(
            f"Error creating ground truth vs generated samples for {dataset_name}: {e}"
        )
        plt.close()
