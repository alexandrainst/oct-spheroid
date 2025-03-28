import argparse
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_best_epoch(tensorboard_file, metric_name):
    event_acc = EventAccumulator(tensorboard_file)
    event_acc.Reload()

    if metric_name not in event_acc.Tags()["scalars"]:
        return None, None

    best_epoch = None
    best_value = float("-inf")

    for scalar in event_acc.Scalars(metric_name):
        if scalar.value > best_value:
            best_value = scalar.value
            best_epoch = scalar.step

    return best_epoch, best_value


def find_event_files(directory):
    event_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))
    return event_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best epoch from all TensorBoard files in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing TensorBoard event files.")
    parser.add_argument("--metric_name", type=str, help="Name of the metric to track.", default="accuracy/val_epoch")
    args = parser.parse_args()

    event_files = find_event_files(args.directory)
    # sort event files by name
    event_files.sort()

    if not event_files:
        print("No TensorBoard event files found.")
    else:
        for event_file in event_files:
            epoch, value = get_best_epoch(event_file, args.metric_name)
            if epoch is not None:
                print(f"File: {event_file}\nBest epoch: {epoch}, Best {args.metric_name}: {value}\n")
            else:
                print(f"File: {event_file}\nNo data found for metric: {args.metric_name}\n")
