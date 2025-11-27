import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

try:
    plt.figure()
    plt.plot(
        experiment_data["input_feature_selection"]["synthetic_dataset"]["losses"][
            "train"
        ],
        label="Training Loss",
    )
    plt.title("Synthetic Dataset - Training Loss")
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
    plt.plot(
        experiment_data["input_feature_selection"]["synthetic_dataset"]["metrics"][
            "train"
        ],
        label="SES",
    )
    plt.title("Synthetic Dataset - Training Scenario Evaluation Score (SES)")
    plt.xlabel("Epochs")
    plt.ylabel("SES Value")
    plt.legend()
    plt.savefig(os.path.join(working_dir, "synthetic_dataset_ses.png"))
    plt.close()
except Exception as e:
    print(f"Error creating SES plot: {e}")
    plt.close()
