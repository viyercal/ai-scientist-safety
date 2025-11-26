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

# Plot training loss
try:
    plt.figure()
    for lr in experiment_data["learning_rate_tuning"]:
        losses = experiment_data["learning_rate_tuning"][lr]["losses"]["train"]
        plt.plot(losses, label=f"LR = {lr}")
    plt.title("Training Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "simple_dynamic_training_loss_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot training accuracy
try:
    plt.figure()
    for lr in experiment_data["learning_rate_tuning"]:
        metrics = experiment_data["learning_rate_tuning"][lr]["metrics"]["train"]
        plt.plot(metrics, label=f"LR = {lr}")
    plt.title("Training Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "simple_dynamic_training_accuracy_curve.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training accuracy plot: {e}")
    plt.close()
