import argparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_best_epoch(tensorboard_file, metric_name):
    event_acc = EventAccumulator(tensorboard_file)
    event_acc.Reload()

    if metric_name not in event_acc.Tags()["scalars"]:
        print(f"Metric {metric_name} not found in TensorBoard file.")
        return None, None

    best_epoch = None
    best_value = float("-inf")

    for scalar in event_acc.Scalars(metric_name):
        if scalar.value > best_value:
            best_value = scalar.value
            best_epoch = scalar.step

    return best_epoch, best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best epoch from a TensorBoard file.")
    parser.add_argument("tensorboard_file", type=str, help="Path to the TensorBoard event file.")
    parser.add_argument("--metric_name", type=str, help="Name of the metric to track.", default="accuracy/val_epoch")
    args = parser.parse_args()

    epoch, value = get_best_epoch(args.tensorboard_file, args.metric_name)

    if epoch is not None:
        print(f"Best epoch: {epoch}, Best {args.metric_name}: {value}")
    else:
        print(f"No data found for metric: {args.metric_name}")
