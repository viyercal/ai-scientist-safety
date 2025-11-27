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

try:
    plt.figure()
    plt.plot(
        experiment_data["dataset_name_1"]["losses"]["train"], label="Training Loss"
    )
    plt.plot(
        experiment_data["dataset_name_1"]["losses"]["val"], label="Validation Loss"
    )
    plt.title("Loss Curve for Dataset: dataset_name_1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "dataset_name_1_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()  # Always close figure even if error occurs
