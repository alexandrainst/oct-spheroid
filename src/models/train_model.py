import os

import numpy as np
import torch
from tqdm import tqdm

# import torchvision
from src.models.generate_model import generate_model
from src.models.multiview import SeparateChannelsResNet18V2

DEBUG = True


def train_multiview(traindataload, valdataload, writer, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device("cuda")
    net = SeparateChannelsResNet18V2(num_classes=config["MODEL"]["NUMBER_OF_CLASSES"], num_views=config["MODEL"]["NUMBER_OF_VIEWS"])
    net = net.float()

    # load pretrained model
    if config["MODEL"]["PRETRAINED"]:
        assert "PRETRAINED_WEIGHTS" in config["MODEL"], "Pretrained weights key not found in config"
        assert os.path.exists(config["MODEL"]["PRETRAINED_WEIGHTS"]), f"Pretrained weights file not found at {config['MODEL']['PRETRAINED_WEIGHTS']}"
        net.load_state_dict(torch.load(config["MODEL"]["PRETRAINED_WEIGHTS"], map_location=device))
    else:
        print("Not using pretrained weights, initializing model with random weights")
    net = net.to(device)

    torch.cuda.empty_cache()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion =  nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["OPTIMIZER"]["LEARNING_RATE"], weight_decay=config["OPTIMIZER"]["WEIGHT_DECAY"])
    if config["MODEL"]["PRETRAINED"]:
        assert "OPTIMIZER_STATE" in config["MODEL"], "Optimizer state key not found in config"
        assert os.path.exists(config["MODEL"]["OPTIMIZER_STATE"]), f"Optimizer state file not found at {config['MODEL']['OPTIMIZER_STATE']}"
        optimizer.load_state_dict(torch.load(config["MODEL"]["OPTIMIZER_STATE"], map_location=device))
    else:
        print("Not using pretrained optimizer state, initializing optimizer with random state")

    total_epochs = config["TRAINING"]["START_EPOCH"] + config["TRAINING"]["EPOCHS"]
    for epoch in range(config["TRAINING"]["START_EPOCH"], total_epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        net.train()
        for step, (img, label) in tqdm(enumerate(traindataload), desc=f"Train epoch {epoch + 1}/{total_epochs}", total=len(traindataload)):
            optimizer.zero_grad()
            # if DEBUG:
            #     import torchvision
            #     # plot image
            #     os.makedirs("./debug_images_2D", exist_ok=True)
            #     for i_view in range(img.shape[1]):
            #         torchvision.utils.save_image(img[:,i_view].unsqueeze(1), f"./debug_images_2D/img_view{i_view}_{step}.png", normalize=True)

            #     print('here')
            img = img.to(device)
            label = label.to(device)
            pred = net.forward(img.float())
            loss = criterion(pred, label)
            acc = accuracy(pred=pred, label=label)
            train_loss.append(loss.item())
            train_acc.append(acc)
            loss.backward()
            optimizer.step()
            log_step = (len(traindataload) * epoch) + step
            writer["writer"].add_scalar("loss/train_step", loss.item(), log_step)
            writer["writer"].add_scalar("accuracy/train_step", acc, log_step)

        with torch.no_grad():
            for step, (img, label) in tqdm(enumerate(valdataload), desc=f"Validation epoch {epoch + 1}/{total_epochs}", total=len(valdataload)):
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

        torch.save(net.state_dict(), writer["run_dir"] + "/model_" + str(config["GENERAL"]["NAME"]) + "_" + str(epoch) + ".pt")
        torch.save(optimizer.state_dict(), writer["run_dir"] + "/optimizer_" + str(config["GENERAL"]["NAME"]) + "_" + str(epoch) + ".pt")


def train3D(traindataload, valdataload, writer, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["GENERAL"]["TASK"] == "classification":
        net = generate_model("resnet", 18, config["MODEL"]["NUMBER_OF_CLASSES"], 1, 128)
        criterion = torch.nn.CrossEntropyLoss()
    elif config["GENERAL"]["TASK"] == "regression":
        net = generate_model("resnet", 18, 1, 1, 128)
        criterion = torch.nn.MSELoss()
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(net.parameters(), lr=config["OPTIMIZER"]["LEARNING_RATE"], weight_decay=config["OPTIMIZER"]["WEIGHT_DECAY"])
    net = net.to(device)
    os.makedirs("./reports/models", exist_ok=True)
    accumulation_steps = config["TRAINING"]["LOSS_ACCUMULATION_STEPS"]
    traindataloader_iter = iter(traindataload)
    total_epochs = config["TRAINING"]["START_EPOCH"] + config["TRAINING"]["EPOCHS"]
    for epoch in range(config["TRAINING"]["START_EPOCH"], total_epochs):
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for step in tqdm(range(len(traindataload)), desc=f"Train epoch {epoch + 1}/{total_epochs}"):
            img, label = next(traindataloader_iter)
            img = img.to(device, dtype=torch.float16)
            net.train()
            label = label.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = net.forward(img)
                # TODO check if we need a activation funciton to limit the output to 0-1 e.g sigmoid or does it harm the model training
                if config["GENERAL"]["TASK"] == "regression":
                    pred = torch.sigmoid(pred)
                    pred = pred[:, 0]
                loss = criterion(pred, label) / accumulation_steps
            loss.backward()
            pred = pred.detach()
            loss.detach_()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(traindataload):
                optimizer.step()
                optimizer.zero_grad()  # reset gradients

            acc = accuracy(pred=pred, label=label) if config["GENERAL"]["TASK"] == "classification" else 0
            train_loss.append(loss.item() * accumulation_steps)
            train_acc.append(acc)

            log_step = (len(traindataload) * epoch) + step
            writer["writer"].add_scalar("loss/train_step", loss.item() * accumulation_steps, log_step)
            writer["writer"].add_scalar("accuracy/train_step", acc, log_step)

        with torch.no_grad():
            valdataloader_iter = iter(valdataload)
            for step in tqdm(range(len(valdataload)), desc=f"Validation epoch {epoch + 1}/{total_epochs}"):
                img, label = next(valdataloader_iter)
                img = img.to(device, dtype=torch.float16)
                net.eval()
                label = label.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = net.forward(img)
                    # TODO check if we need a activation funciton to limit the output to 0-1 e.g sigmoid or does it harm the model training
                    if config["GENERAL"]["TASK"] == "regression":
                        pred = torch.sigmoid(pred)
                        pred = pred[:, 0]
                    loss = criterion(pred, label)
                acc = accuracy(pred=pred, label=label) if config["GENERAL"]["TASK"] == "classification" else 0
                val_loss.append(loss.item())
                val_acc.append(acc)
                log_step = (len(valdataload) * epoch) + step
                writer["writer"].add_scalar("loss/val_step", loss.item(), log_step)
                writer["writer"].add_scalar("accuracy/val_step", acc, log_step)

        # save model
        torch.save(net.state_dict(), writer["run_dir"] + "/model_" + str(config["GENERAL"]["NAME"]) + "_" + str(epoch) + ".pt")
        torch.save(optimizer.state_dict(), writer["run_dir"] + "/optimizer_" + str(config["GENERAL"]["NAME"]) + "_" + str(epoch) + ".pt")
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
