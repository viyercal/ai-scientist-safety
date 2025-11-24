import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot training losses for each learning rate
for lr in experiment_data["hyperparam_tuning_learning_rate"]:
    try:
        plt.figure()
        losses = experiment_data["hyperparam_tuning_learning_rate"][lr]["losses"][
            "train"
        ]
        plt.plot(losses, label="Training Loss")
        plt.title(f"Training Loss for Learning Rate: {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"training_loss_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plots for training loss with lr {lr}: {e}")
        plt.close()

# Plot prediction accuracy for each learning rate
for lr in experiment_data["hyperparam_tuning_learning_rate"]:
    try:
        accuracy = experiment_data["hyperparam_tuning_learning_rate"][lr]["metrics"][
            "train"
        ]
        plt.figure()
        plt.plot(accuracy, label="Prediction Accuracy")
        plt.title(f"Prediction Accuracy for Learning Rate: {lr}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"prediction_accuracy_lr_{lr}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plots for prediction accuracy with lr {lr}: {e}")
        plt.close()
