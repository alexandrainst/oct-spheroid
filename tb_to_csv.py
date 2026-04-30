import argparse
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def write_to_csv(tensorboard_file, metric_name, output_file):
    event_acc = EventAccumulator(tensorboard_file)
    event_acc.Reload()

    if metric_name not in event_acc.Tags()["scalars"]:
        print(f"Metric {metric_name} not in {tensorboard_file}.")
        return

    with open(output_file, "w") as f:
        f.write("epoch,value\n")
        for scalar in event_acc.Scalars(metric_name):
            f.write(f"{scalar.step}, {scalar.value}\n")


def find_event_files(directory):
    event_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))
    return event_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best epoch from all TensorBoard files in a directory.")
    # parser.add_argument("--directory", type=str, help="Path to the directory containing TensorBoard event files.", default="runs/3D_20250403-121709")
    # parser.add_argument("--metric_name", type=str, help="Name of the metric to track.", default="accuracy/val_epoch")
    directories = [
        "runs/2D_20250320-093742",
        "runs/3D_20250327-091234"
        # "runs/2D_20250320-141935",
        # "runs/3D_20250403-121709",
    ]
    metric_names = [
        "loss/train_epoch",
        "accuracy/train_epoch",
        "loss/val_epoch",
        "accuracy/val_epoch",
        # Add more metric names as needed
    ]
    args = parser.parse_args()

    event_files = []
    for directory in directories:
        event_files += find_event_files(directory)
    event_files.sort()

    if not event_files:
        print("No TensorBoard event files found.")
    else:
        for event_file, directory in zip(event_files, directories):
            os.makedirs(os.path.join(directory, "csv"), exist_ok=True)
            for metric_name in metric_names:
                output_file = os.path.join(directory, "csv" f"{os.path.basename(event_file)}_{metric_name.replace('/', '_')}.csv")
                write_to_csv(event_file, metric_name, output_file)
                print(f"Exported {metric_name} from {event_file} to {output_file}")
