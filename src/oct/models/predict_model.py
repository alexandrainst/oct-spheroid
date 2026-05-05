import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from oct.features.utils import SeparateChannelsResNet18
from oct.models.generate_model import generate_model


def evaluate(testDataLoader, weights):
    # testDataLoader = DataLoader(testdataset, batch_size=1, shuffle=True)
    acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     device = torch.device('cpu')
    net = generate_model("resnet", 10, 3, 1, 128)  # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    net.load_state_dict(torch.load(weights))
    net = net.to(device)
    labels = []
    preds = []
    with open("./reports/models/evaluation_log_" + str(weights.split("/")[-1].split(".")[0]) + ".txt", "w") as w:
        with torch.no_grad():
            testDataLoader_iter = iter(testDataLoader)
            # for step, (img, label) in enumerate(valdataload):
            for step in tqdm(range(len(testDataLoader)), desc="Test evaluation", leave=True, position=0, total=len(testDataLoader)):
                img, label = next(testDataLoader_iter)
                img = img.to(device, dtype=torch.float16)
                net.train()
                label = label.to(device)
                # img = torch.unsqueeze(img, 0)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = net.forward(img)

                # for step, (img, label) in tqdm(enumerate(testDataLoader), desc="Test evaluation", leave=True, position=0, total=len(testDataLoader)):
                #     # filename = testdataset.df.iloc[step, 0].split("/")[-1]
                #     net.eval()
                #     #         sub_batch = 0
                #     #         for img in batch:
                #     img = img.to(device, dtype=torch.float)

                #     pred = net.forward(img)
                single_acc = accuracy(pred=pred, label=label)
                preds.append(torch.max(pred, 1)[1].item())
                labels.append(label.item())
                w.write(
                    f"True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred, 1)[1].item()} with probability {pred}. \n"
                )  # for a volume {filename}
                # print(f'True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()}')
                acc.append(single_acc)
        w.write(f"Mean accuracy across test dataset: {np.mean(acc)} \n")
        print(f"Mean accuracy across test dataset: {np.mean(acc)}")

    confusion_matrix = metrics.confusion_matrix(labels, preds)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["CTRL", "25", "100"])
    plt.figure(figsize=(20, 20))
    cm_display.plot()
    plt.show()
    plt.savefig("./reports/figures/evaluation_confusion_matrix_" + str(weights.split("/")[-1].split(".")[0]) + ".png")


def evaluate2D(testdataset, weights, n_classes):
    testDataLoader = DataLoader(testdataset, batch_size=1, shuffle=True)
    acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     device = torch.device('cpu')
    #     net = torchvision.models.resnet50(
    #             weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    #         )
    #     num_texture_classes = 2
    #     num_features = net.fc.in_features
    #     net.fc = nn.Sequential(
    #             nn.Linear(num_features, 1024),
    #             nn.ReLU(),
    #             nn.Linear(1024, num_texture_classes),
    #         )
    #     new_first_layer = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # # Replace the original first layer with the new one
    #     net.conv1 = new_first_layer

    net = SeparateChannelsResNet18(num_classes=n_classes, num_views=4)
    net = net.float()
    net.load_state_dict(torch.load(weights))
    net = net.to(device)
    labels = []
    preds = []
    with open("./reports/models/evaluation_log_" + str(weights.split("/")[-1].split(".")[0]) + "_shuffle.txt", "w") as w:
        with torch.no_grad():
            for step, (img, label, filename) in tqdm(enumerate(testDataLoader), desc="Test evaluation", leave=True, position=0, total=len(testDataLoader)):
                # filename = testdataset.df.iloc[step, 1]
                net.train()
                #         sub_batch = 0
                #         for img in batch:
                img = img.to(device, dtype=torch.float)
                pred = net.forward(img)
                single_acc = accuracy(pred=pred, label=label)
                preds.append(torch.max(pred, 1)[1].item())
                labels.append(label.item())
                w.write(f"True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred, 1)[1].item()} for a volume {filename}. \n")
                # print(f'True label for this volume: {label.item()}, prediction for this volume: {torch.max(pred,1)[1].item()}')
                acc.append(single_acc)
        w.write(f"Mean accuracy across test dataset: {np.mean(acc)} \n")
        print(f"Mean accuracy across test dataset: {np.mean(acc)}")

    confusion_matrix = metrics.confusion_matrix(labels, preds)
    if n_classes == 3:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["CTRL", "25", "100"])
    else:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    plt.figure(figsize=(20, 20))
    cm_display.plot()
    plt.show()
    plt.savefig("./reports/figures/evaluation_confusion_matrix_" + str(weights.split("/")[-1].split(".")[0]) + ".png")


def accuracy(pred, label):
    # print(pred.shape,label.shape)
    pred = pred.cpu()
    label = label.cpu()
    corr = 0
    for i in range(pred.shape[0]):
        _, p = torch.max(pred, 1)
        if p[i].item() == label[i].item():
            corr = corr + 1
    score = (corr / pred.shape[0]) * 100
    return score
