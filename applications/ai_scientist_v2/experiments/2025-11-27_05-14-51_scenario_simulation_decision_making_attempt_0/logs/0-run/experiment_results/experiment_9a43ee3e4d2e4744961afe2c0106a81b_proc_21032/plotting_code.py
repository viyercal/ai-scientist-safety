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

for dataset_name, data in experiment_data["multi_dataset_evaluation"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.plot(data["losses"]["val"], label="Validation Loss")
        plt.title(f"{dataset_name}: Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{working_dir}/{dataset_name}_loss_curve.png")
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {dataset_name}: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="Training SORS")
        plt.plot(data["metrics"]["val"], label="Validation SORS")
        plt.title(f"{dataset_name}: Training and Validation SORS")
        plt.xlabel("Epoch")
        plt.ylabel("SORS")
        plt.legend()
        plt.savefig(f"{working_dir}/{dataset_name}_sors_curve.png")
        plt.close()
    except Exception as e:
        print(f"Error creating SORS plot for {dataset_name}: {e}")
        plt.close()
