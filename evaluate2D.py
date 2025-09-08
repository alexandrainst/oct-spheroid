import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from src.data.dataset_multiview import MultiviewDataset
from src.models.multiview import SeparateChannelsResNet18
from src.models.train_model import accuracy
from src.utils.config import config_handler

if __name__ == "__main__":
    # read yaml file
    parser = argparse.ArgumentParser(description="2D Model Evaluation")
    parser.add_argument("config", type=str, default="config/experiments/config_2D.yaml", help="Path to the config file")
    parser.add_argument("weights", type=str, help="Path to the weights file")
    parser.add_argument
    args = parser.parse_args()
    config = config_handler(args.config)

    transforms_val = v2.Compose(
        [
            v2.Normalize(mean=[12.4361, 12.4864, 12.4278, 12.3149], std=[10.7506, 10.7760, 9.8911, 9.9736]),
        ]
    )

    test_df = pd.read_csv("./data/test_no_outliers.csv")
    testdataset = MultiviewDataset(df=test_df, n_classes=3, return_filename=True, transform=transforms_val, labels_dict={"CTRL": 0, "25": 1, "100": 2})
    testdataloader = DataLoader(testdataset, batch_size=1)
    acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = SeparateChannelsResNet18(num_classes=config["MODEL"]["NUMBER_OF_CLASSES"], num_views=config["MODEL"]["NUMBER_OF_VIEWS"])
    # net = net.float()
    net.load_state_dict(torch.load(args.weights, map_location=device))
    net = net.to(device)
    labels = []
    preds = []
    # base_path of weight
    base_path = os.path.dirname(args.weights)
    eval_path = os.path.join(base_path, "evaluation")
    os.makedirs(eval_path, exist_ok=True)

    with open(os.path.join(eval_path, "evaluation_log.txt"), "w") as w:
        with torch.no_grad():
            for step, (img, label, filename) in tqdm(enumerate(testdataloader), desc="Test evaluation", leave=True, position=0, total=len(testdataloader)):
                net.eval()  # TODO change to eval if models are more generalized
                img = img.to(device, dtype=torch.float)
                pred = net.forward(img)
                single_acc = accuracy(pred=pred, label=label)
                preds.append(torch.argmax(pred).item())
                labels.append(label.item())
                w.write(f"True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()} for a volume {filename}. \n")
                # print(f'True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()}')
                acc.append(single_acc)
        w.write(f"Mean accuracy across test dataset:{np.mean(acc)} \n")
        print(f"Mean accuracy across test dataset:{np.mean(acc)}")

    confusion_matrix = metrics.confusion_matrix(labels, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1, 2])
    plt.figure(figsize=(20, 20))
    cm_display.plot(im_kw={"vmin": 0, "vmax": 20})
    plt.savefig(os.path.join(eval_path, "evaluation_confusion_matrix.png"))
    plt.clf()
    plt.close()
