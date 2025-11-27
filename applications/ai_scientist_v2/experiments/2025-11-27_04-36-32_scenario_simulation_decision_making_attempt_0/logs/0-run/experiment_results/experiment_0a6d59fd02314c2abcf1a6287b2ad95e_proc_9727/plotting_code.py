import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
experiment_data = np.load(
    os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
).item()

# Plot losses and metrics for each dataset
for dataset_name in ["dataset_1", "dataset_2", "dataset_3"]:
    try:
        plt.figure()
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "train"
            ],
            label="Train Loss",
        )
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][dataset_name]["losses"][
                "val"
            ],
            label="Validation Loss",
        )
        plt.title(f"{dataset_name} Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {dataset_name} loss plot: {e}")
        plt.close()

    try:
        plt.figure()
        plt.plot(
            experiment_data["multiple_synthetic_datasets"][dataset_name]["metrics"][
                "train"
            ],
            label="SES Train",
        )
        plt.title(f"{dataset_name} SES Metric")
        plt.xlabel("Epochs")
        plt.ylabel("SES")
        plt.legend()
        plt.savefig(os.path.join(working_dir, f"{dataset_name}_ses_metric.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating {dataset_name} SES plot: {e}")
        plt.close()
