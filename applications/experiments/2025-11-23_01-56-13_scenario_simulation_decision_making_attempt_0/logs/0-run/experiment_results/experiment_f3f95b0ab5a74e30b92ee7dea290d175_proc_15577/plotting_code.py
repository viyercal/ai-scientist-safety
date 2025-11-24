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

# Plotting training loss
try:
    plt.figure()
    plt.plot(
        experiment_data["batch_size_tuning"]["simple_dynamic"]["losses"]["train"],
        label="Training Loss",
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "simple_dynamic_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for training loss: {e}")
    plt.close()

# Plotting training accuracy
try:
    plt.figure()
    plt.plot(
        experiment_data["batch_size_tuning"]["simple_dynamic"]["metrics"]["train"],
        label="Training Accuracy",
    )
    plt.title("Training Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "simple_dynamic_training_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot for training accuracy: {e}")
    plt.close()
