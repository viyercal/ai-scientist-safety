import matplotlib.pyplot as plt
import numpy as np
import os

# Set working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training and validation losses for each learning rate
for lr in experiment_data["hyperparam_tuning_learning_rate"]:
    train_losses = experiment_data["hyperparam_tuning_learning_rate"][lr]["losses"][
        "train"
    ]
    val_losses = experiment_data["hyperparam_tuning_learning_rate"][lr]["losses"]["val"]

    try:
        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title(f"{lr} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{lr}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for {lr}: {e}")
        plt.close()

# Plot SORS metrics for each learning rate
for lr in experiment_data["hyperparam_tuning_learning_rate"]:
    train_metrics = experiment_data["hyperparam_tuning_learning_rate"][lr]["metrics"][
        "train"
    ]
    val_metrics = experiment_data["hyperparam_tuning_learning_rate"][lr]["metrics"][
        "val"
    ]

    try:
        plt.figure()
        plt.plot(train_metrics, label="Training SORS")
        plt.plot(val_metrics, label="Validation SORS")
        plt.title(f"{lr} SORS Curves")
        plt.xlabel("Epochs")
        plt.ylabel("SORS")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{lr}_sors_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SORS plot for {lr}: {e}")
        plt.close()
