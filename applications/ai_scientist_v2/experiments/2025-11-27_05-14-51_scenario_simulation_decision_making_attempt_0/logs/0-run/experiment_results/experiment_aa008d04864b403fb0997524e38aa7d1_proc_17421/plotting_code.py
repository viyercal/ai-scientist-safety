import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    train_losses = experiment_data["hyperparam_tuning_batch_size"]["synthetic_dataset"][
        "losses"
    ]["train"]
    val_losses = experiment_data["hyperparam_tuning_batch_size"]["synthetic_dataset"][
        "losses"
    ]["val"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_loss_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    train_metrics = experiment_data["hyperparam_tuning_batch_size"][
        "synthetic_dataset"
    ]["metrics"]["train"]
    val_metrics = experiment_data["hyperparam_tuning_batch_size"]["synthetic_dataset"][
        "metrics"
    ]["val"]
    plt.figure()
    plt.plot(train_metrics, label="Training SORS")
    plt.plot(val_metrics, label="Validation SORS")
    plt.title("SORS Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("SORS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_sors_plot.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SORS plot: {e}")
    plt.close()
