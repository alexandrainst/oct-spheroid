import os

import numpy as np
import torch
from tqdm import tqdm

# import torchvision
from src.models.generate_model import generate_model
from src.models.multiview import SeparateChannelsResNet18


def train_multiview(traindataload, valdataload, model_name, epochs, patience, learning_rate, n_classes, writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device("cuda")
    net = SeparateChannelsResNet18(num_classes=n_classes, num_views=4)
    net = net.float()

    torch.cuda.empty_cache()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion =  nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)
    accum_iter = 6
    train_loss_final = []
    train_acc_final = []
    val_loss_final = []
    val_acc_final = []
    for epoch in range(epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for step, (img, label) in tqdm(enumerate(traindataload)):
            net.train()
            img = img.to(device, dtype=torch.float)
            label = label.to(device)
            optimizer.zero_grad()
            pred = net.forward(img.float())
            loss = criterion(pred, label)
            loss = loss / accum_iter
            acc = accuracy(pred=pred, label=label)
            train_loss.append(loss.item())
            train_acc.append(acc)
            loss.backward()
            optimizer.step()
            log_step = (len(traindataload) * epoch) + step
            writer["writer"].add_scalar("loss/train_step", loss.item(), log_step)
            writer["writer"].add_scalar("accuracy/train_step", acc, log_step)

        with torch.no_grad():
            for step, (img, label) in enumerate(valdataload):
                net.eval()
                img = img.to(device, dtype=torch.float)
                label = label.to(device)
                pred = net.forward(img.float())
                loss = criterion(pred, label)
                acc = accuracy(pred=pred, label=label)
                val_loss.append(loss.item())
                val_acc.append(acc)
                log_step = (len(valdataload) * epoch) + step
                writer["writer"].add_scalar("loss/val_step", loss.item(), log_step)
                writer["writer"].add_scalar("accuracy/val_step", acc, log_step)

        writer["writer"].add_scalar("loss/train_epoch", np.mean(train_loss), epoch)
        writer["writer"].add_scalar("accuracy/train_epoch", np.mean(train_acc), epoch)
        writer["writer"].add_scalar("loss/val_epoch", np.mean(val_loss), epoch)
        writer["writer"].add_scalar("accuracy/val_epoch", np.mean(val_acc), epoch)

        torch.save(net.state_dict(), writer["run_dir"] + "/model_" + str(model_name) + "_" + str(epoch) + ".pt")

        train_loss_final.append(np.mean(train_loss))
        train_acc_final.append(np.mean(train_acc))
        val_loss_final.append(np.mean(val_loss))
        val_acc_final.append(np.mean(val_acc))


def train3D(traindataload, valdataload, model_name, epochs, patience, learning_rate, writer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = generate_model("resnet", 18, 3, 1, 128)

    torch.cuda.empty_cache()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)
    accum_iter = 6

    os.makedirs("./reports/models", exist_ok=True)

    traindataloader_iter = iter(traindataload)
    for epoch in range(epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for step in tqdm(range(len(traindataload)), desc=f"Train epoch {epoch+1}/{epochs}"):
            img, label = next(traindataloader_iter)
            img = img.to(device, dtype=torch.float16)
            net.train()
            label = label.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = net.forward(img)
                loss = criterion(pred, label)
            loss = loss / accum_iter
            acc = accuracy(pred=pred, label=label)
            train_loss.append(loss.item())
            train_acc.append(acc)
            loss.backward()
            optimizer.step()
            log_step = (len(traindataload) * epoch) + step
            writer["writer"].add_scalar("loss/train_step", loss.item(), log_step)
            writer["writer"].add_scalar("accuracy/train_step", acc, log_step)

        with torch.no_grad():
            valdataloader_iter = iter(valdataload)
            for step in tqdm(range(len(valdataload)), desc=f"Validation epoch {epoch+1}/{epochs}"):
                img, label = next(valdataloader_iter)
                img = img.to(device, dtype=torch.float16)
                net.eval()
                label = label.to(device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = net.forward(img)
                    loss = criterion(pred, label)
                acc = accuracy(pred=pred, label=label)
                val_loss.append(loss.item())
                val_acc.append(acc)
                log_step = (len(valdataload) * epoch) + step
                writer["writer"].add_scalar("loss/val_step", loss.item(), log_step)
                writer["writer"].add_scalar("accuracy/val_step", acc, log_step)

        # save model
        torch.save(net.state_dict(), writer["run_dir"] + "/model_" + str(model_name) + "_" + str(epoch) + ".pt")

        writer["writer"].add_scalar("loss/train_epoch", np.mean(train_loss), epoch)
        writer["writer"].add_scalar("accuracy/train_epoch", np.mean(train_acc), epoch)
        writer["writer"].add_scalar("loss/val_epoch", np.mean(val_loss), epoch)
        writer["writer"].add_scalar("accuracy/val_epoch", np.mean(val_acc), epoch)


def accuracy(pred, label):
    pred = pred.cpu()
    label = label.cpu()
    corr = 0
    for i in range(pred.shape[0]):
        _, p = torch.max(pred, 1)
        if p[i].item() == label[i].item():
            corr = corr + 1
    score = (corr / pred.shape[0]) * 100
    return score
