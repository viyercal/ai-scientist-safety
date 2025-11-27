import matplotlib.pyplot as plt
import numpy as np
import os

# Define working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot training loss
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_dataset"]["losses"]["train"], label="Training Loss"
    )
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

# Plot SES
try:
    plt.figure()
    plt.plot(
        experiment_data["synthetic_dataset"]["metrics"]["train"],
        label="Scenario Evaluation Score (SES)",
    )
    plt.title("SES Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("SES")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_ses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SES plot: {e}")
    plt.close()
