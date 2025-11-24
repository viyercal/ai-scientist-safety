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

for lr in experiment_data["hyperparam_tuning"]["learning_rate"]:
    try:
        train_losses = experiment_data["hyperparam_tuning"]["learning_rate"][lr][
            "losses"
        ]["train"]
        val_losses = experiment_data["hyperparam_tuning"]["learning_rate"][lr][
            "losses"
        ]["val"]
        epochs = range(1, len(train_losses) + 1)

        plt.figure()
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title(f"Learning Rate: {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.suptitle("Loss Curves for Dynamic Environment Dataset")
        plt.savefig(os.path.join(working_dir, f"loss_curves_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot for learning rate {lr}: {e}")
        plt.close()
