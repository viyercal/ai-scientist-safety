import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plotting training loss
for batch_size, data in experiment_data["batch_size_tuning"].items():
    try:
        plt.figure()
        plt.plot(data["losses"]["train"], label="Training Loss")
        plt.title(f"Training Loss for Batch Size {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_batch_{batch_size}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot for batch size {batch_size}: {e}")
        plt.close()

# Plotting Scenario Evaluation Score
for batch_size, data in experiment_data["batch_size_tuning"].items():
    try:
        plt.figure()
        plt.plot(data["metrics"]["train"], label="SES")
        plt.title(f"Scenario Evaluation Score for Batch Size {batch_size}")
        plt.xlabel("Epochs")
        plt.ylabel("SES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"ses_batch_{batch_size}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating SES plot for batch size {batch_size}: {e}")
        plt.close()
