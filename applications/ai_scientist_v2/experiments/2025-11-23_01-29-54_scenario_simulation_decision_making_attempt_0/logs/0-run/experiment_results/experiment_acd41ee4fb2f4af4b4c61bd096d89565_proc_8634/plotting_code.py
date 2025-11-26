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
    train_losses = experiment_data["hyperparam_tuning_learning_rate"]["synthetic_data"][
        "losses"
    ]["train"]
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

try:
    train_metrics = experiment_data["hyperparam_tuning_learning_rate"][
        "synthetic_data"
    ]["metrics"]["train"]
    plt.figure()
    plt.plot(train_metrics, label="Training Metric (SCS)")
    plt.title("Training Metric Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("SCS")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_data_training_metric.png"))
    plt.close()
except Exception as e:
    print(f"Error creating metric plot: {e}")
    plt.close()
