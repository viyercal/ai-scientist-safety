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
    epochs = list(
        range(1, len(experiment_data["synthetic_data"]["losses"]["train"]) + 1)
    )
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_data"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    plt.figure()
    plt.plot(
        epochs,
        experiment_data["synthetic_data"]["metrics"]["train"],
        label="Training Metric",
    )
    plt.title("Training Metric over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training metric plot: {e}")
    plt.close()
