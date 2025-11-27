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

# Plotting training losses
try:
    epochs = np.arange(1, 51)
    train_losses = experiment_data["synthetic_data"]["losses"]["train"]
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(working_dir, "training_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plotting validation losses
try:
    val_losses = experiment_data["synthetic_data"]["losses"]["val"]
    plt.figure()
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
    plt.title("Validation Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(working_dir, "validation_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating validation loss plot: {e}")
    plt.close()

# Plotting SCS metrics for train and val
try:
    train_scs = experiment_data["synthetic_data"]["metrics"]["train"]
    val_scs = experiment_data["synthetic_data"]["metrics"]["val"]
    plt.figure()
    plt.plot(epochs, train_scs, label="Train SCS")
    plt.plot(epochs, val_scs, label="Validation SCS", color="red")
    plt.title("Metrics Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("SCS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "metrics_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SCS plot: {e}")
    plt.close()
