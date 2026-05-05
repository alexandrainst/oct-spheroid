import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from oct.data.dataset_3D import CancerSphere3DDataset
from oct.models.generate_model import generate_model
from oct.models.train_model import accuracy
from oct.utils.config import config_handler

if __name__ == "__main__":
    # read yaml file
    parser = argparse.ArgumentParser(description="3D Model Training")
    parser.add_argument("config", type=str, default="config/experiments/config_3D.yaml", help="Path to the config file")
    parser.add_argument("weights", type=str, help="Path to the weights file")
    parser.add_argument
    args = parser.parse_args()
    config = config_handler(args.config)
    labels_dict = {
        "CTRL": 0,
        "25": 1,
        "100": 2,
    }
    test_df = pd.read_csv("./data/test_no_outliers.csv")
    testdataset = CancerSphere3DDataset(
        df=test_df,
        n_classes=3,
        reshape_size=(200, 500, 500),
        task=config["GENERAL"]["TASK"],
        labels_dict=labels_dict,
        return_filename=True,
        mean=config["DATASET"]["MEAN"],
        std=config["DATASET"]["STD"],
        normalize=False,
    )
    testdataloader = DataLoader(testdataset, batch_size=1)

    testdataset.mean, testdataset.std

    acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.cuda.is_available(), "CUDA not available"

    net = generate_model("resnet", 18, 3, 1, 128)
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
                net.train()  # TODO change to eval if models are more generalized
                # net.eval()
                img = img.to(device, dtype=torch.float)
                # normalize
                img = img.sub(testdataset.mean[0]).div(testdataset.std[0])
                with torch.amp.autocast(dtype=torch.float16, device_type=device.type):
                    pred = net.forward(img)
                single_acc = accuracy(pred=pred, label=label)
                preds.append(torch.argmax(pred).item())
                labels.append(label.item())
                w.write(f"True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred, 1)[1].item()} for a volume {filename}. \n")
                # print(f'True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()}')
                acc.append(single_acc)
        w.write(f"Mean accuracy across test dataset: {np.mean(acc)} \n")
        print(f"Mean accuracy across test dataset: {np.mean(acc)}")

    confusion_matrix = metrics.confusion_matrix(labels, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1, 2])
    plt.figure(figsize=(20, 20))
    cm_display.plot(im_kw={"vmin": 0, "vmax": 20})
    plt.savefig(os.path.join(eval_path, "evaluation_confusion_matrix.png"))
    plt.clf()
    plt.close()
