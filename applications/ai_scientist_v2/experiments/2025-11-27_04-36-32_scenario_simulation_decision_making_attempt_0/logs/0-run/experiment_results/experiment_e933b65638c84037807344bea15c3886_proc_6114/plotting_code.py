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
    train_losses = experiment_data["hyperparam_tuning_learning_rate"][
        "synthetic_dataset"
    ]["losses"]["train"]
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_training_loss.png"))
    plt.close()
except Exception as e:
    print(f"Error creating training loss plot: {e}")
    plt.close()

try:
    plt.figure()
    train_metrics = experiment_data["hyperparam_tuning_learning_rate"][
        "synthetic_dataset"
    ]["metrics"]["train"]
    plt.plot(train_metrics, label="Scenario Evaluation Score (SES)", color="orange")
    plt.title("SES over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("SES Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_ses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SES plot: {e}")
    plt.close()
