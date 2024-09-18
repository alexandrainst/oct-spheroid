import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# import torchvision
from src.models.generate_model import main
from src.features.utils import SeparateChannelsResNet18

random.seed(45)


def train_multiview(traindataload, valdataload, model_name, epochs, patience, learning_rate,n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     device = torch.device('cpu')
    # net = main('resnet',10,2,1,128)    # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify the last layer of the model to match the number of texture classes in DTD

    # net = torchvision.models.resnet50(
    #         weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    #     )
    # num_texture_classes = 2
    # num_features = net.fc.in_features
    # net.fc = nn.Sequential(
    #         nn.Linear(num_features, 1024),
    #         nn.ReLU(),
    #         nn.Linear(1024, num_texture_classes),
    #     )

    # fl = new_first_layer(4)
    # net.conv1 = fl

    net = SeparateChannelsResNet18(num_classes=n_classes, num_views=4)
    net = net.float()

    torch.cuda.empty_cache()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion =  nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)
    # net = net.float()
    # epochs = 1000
    accum_iter = 6
    best_loss = 0
    counter = 0
    train_loss_final = []
    train_acc_final = []
    val_loss_final = []
    val_acc_final = []
    with open("./reports/models/training_log_" + str(model_name) + ".txt", "w") as w:
        for epoch in range(epochs):
            start_time_epoch = time.time()
            train_loss = []
            train_acc = []
            val_loss = []
            val_acc = []
            for step, (img, label) in enumerate(traindataload):
                start_time_step = time.time()
                net.train()
                #         sub_batch = 0
                #         for img in batch:
                img = img.to(device, dtype=torch.float)
                # img = img.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                pred = net.forward(img.float())
                # print(pred,label)
                #             print(pred.shape,label.shape)
                loss = criterion(pred, label)
                loss = loss / accum_iter
                acc = accuracy(pred=pred, label=label)
                train_loss.append(loss.item())
                train_acc.append(acc)
                loss.backward()
                # if ((step + 1) % accum_iter == 0) or (step + 1 == len(traindataload)):
                #     optimizer.step()
                #     optimizer.zero_grad()
                optimizer.step()
                end_time_step = time.time()
                #             sub_batch += 1
                print(
                    "Epoch [{}/{}], Step [{}/{}], training loss:{:.4f}, time passed:{}".format(
                        epoch + 1, epochs, step + 1, len(traindataload), loss.item(), end_time_step - start_time_step
                    )
                )
                w.write(
                    "Epoch [{}/{}], Step [{}/{}], training loss:{:.4f}, time passed:{} \n".format(
                        epoch + 1, epochs, step + 1, len(traindataload), loss.item(), end_time_step - start_time_step
                    )
                )
            with torch.no_grad():
                for step, (img, label) in enumerate(valdataload):
                    #             sub_batch = 0
                    #             for img in batch:
                    start_time_step = time.time()
                    net.eval()
                    img = img.to(device, dtype=torch.float)
                    # img = img.to(device)
                    label = label.to(device)
                    pred = net.forward(img.float())
                    loss = criterion(pred, label)
                    acc = accuracy(pred=pred, label=label)
                    val_loss.append(loss.item())
                    val_acc.append(acc)
                    end_time_step = time.time()
                    #                 sub_batch += 1
                    print(
                        "Epoch [{}/{}], Step [{}/{}], val loss:{:.4f}, time passed:{} ".format(
                            epoch + 1, epochs, step + 1, len(valdataload), loss.item(), end_time_step - start_time_step
                        )
                    )
                    w.write(
                        "Epoch [{}/{}], Step [{}/{}], val loss:{:.4f}, time passed:{} \n".format(
                            epoch + 1, epochs, step + 1, len(valdataload), loss.item(), end_time_step - start_time_step
                        )
                    )

            end_time_epoch = time.time()
            # print('After Epoch [{}/{}], average training loss:{:.4f}, average val loss:{:.4f}'.format(epoch+1,epochs,np.mean(train_loss),np.mean(val_loss)))
            print(
                "After Epoch [{}/{}], average training loss:{:.4f}, average training accuracy {:.4f}, average val loss:{:.4f}, average val {:.4f}, time passed:{}".format(
                    epoch + 1, epochs, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc), end_time_epoch - start_time_epoch
                )
            )

            # w.write('After Epoch [{}/{}], average training loss:{:.4f}, average val loss:{:.4f} \n'.format(epoch+1,epochs,np.mean(train_loss),np.mean(val_loss)))
            w.write(
                " After Epoch [{}/{}], average training loss:{:.4f}, average training accuracy {:.4f}, average val loss:{:.4f}, average val {:.4f}, time passed:{} \n".format(
                    epoch + 1, epochs, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc), end_time_epoch - start_time_epoch
                )
            )

            current_loss = np.mean(val_loss)
            if best_loss == 0:
                best_loss = current_loss
                print("FIRST EPOCH!!")
                w.write("FIRST EPOCH!! \n")
                torch.save(net.state_dict(), "./models/model_" + str(model_name) + ".pt")
            elif current_loss < best_loss:
                print("GOOD EPOCH!!")
                w.write("GOOD EPOCH!! \n")
                best_loss = current_loss
                torch.save(net.state_dict(), "./models/model_" + str(model_name) + ".pt")
                counter = 0
            else:
                print("BAD EPOCH!!!")
                w.write("BAD EPOCH!! \n")
                counter += 1
                if counter >= patience:
                    print("MODEL STARTS TO OVERFIT!!!")
                    w.write("MODEL STARTS TO OVERFIT!!! \n")
                    break
            train_loss_final.append(np.mean(train_loss))
            train_acc_final.append(np.mean(train_acc))
            val_loss_final.append(np.mean(val_loss))
            val_acc_final.append(np.mean(val_acc))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))
        ax1.plot(train_loss_final, label="Training loss")
        ax1.plot(val_loss_final, label="Validation loss")
        ax1.legend(loc="upper right")
        ax2.plot(train_acc_final, label="Training accuracy")
        ax2.plot(val_acc_final, label="Validation accuracy")
        ax2.legend(loc="lower right")
        ax2.set_ylim([0, 110])

        # plt.figure(figsize=(30,20))
        # plt.plot(train_loss_final,label = 'Training loss')
        # plt.plot(val_loss_final,label='Validation loss')
        # plt.legend(loc = 'upper right')
        plt.savefig("./reports/models/learning_curves_" + str(model_name) + ".png")


def train3D(traindataload, valdataload, model_name, epochs, patience, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     device = torch.device('cpu')
    net = main('resnet',10,3,1,128)    # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify the last layer of the model to match the number of texture classes in DTD

    # net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    # num_texture_classes = 2
    # num_features = net.fc.in_features
    # net.fc = nn.Sequential(
    #     nn.Linear(num_features, 1024),
    #     nn.ReLU(),
    #     nn.Linear(1024, num_texture_classes),
    # )
    # new_first_layer = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # # Replace the original first layer with the new one
    # net.conv1 = new_first_layer
    # net.fc = nn.Sequential(
    # nn.Linear(num_features, 1024),
    # nn.ReLU(),
    # nn.Linear(1024, 3)
    # )

    torch.cuda.empty_cache()
    weights = torch.tensor([1.0, 0.9, 0.7], dtype=torch.float16)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion =  nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)
    # net = net.float()
    # epochs = 1000
    accum_iter = 6
    best_loss = 0
    counter = 0
    train_loss_final = []
    train_acc_final = []
    val_loss_final = []
    val_acc_final = []
    with open("./reports/models/training_log_" + str(model_name) + ".txt", "w") as w:
        traindataloader_iter = iter(traindataload)
        for epoch in range(epochs):
            start_time_epoch = time.time()
            train_loss = []
            train_acc = []
            val_loss = []
            val_acc = []
            # for step, (img, label) in enumerate(traindataload):
            for step in range(len(traindataload)):
                start_time_step = time.time()
                img, label = next(traindataloader_iter)
                img = img.to(device, dtype=torch.float16)
                net.train()
                #         sub_batch = 0
                #         for img in batch:
                
                # img = img.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = net.forward(img)
                    # print(pred,label)
                    #             print(pred.shape,label.shape)
                    loss = criterion(pred, label)
                loss = loss / accum_iter
                acc = accuracy(pred=pred, label=label)
                train_loss.append(loss.item())
                train_acc.append(acc)
                loss.backward()
                # if ((step + 1) % accum_iter == 0) or (step + 1 == len(traindataload)):
                #     optimizer.step()
                #     optimizer.zero_grad()
                optimizer.step()
                end_time_step = time.time()
                #             sub_batch += 1
                print(
                    "Epoch [{}/{}], Step [{}/{}], training loss:{:.4f}, time passed:{}".format(
                        epoch + 1, epochs, step + 1, len(traindataload), loss.item(), end_time_step - start_time_step
                    )
                )
                w.write(
                    "Epoch [{}/{}], Step [{}/{}], training loss:{:.4f}, time passed:{} \n".format(
                        epoch + 1, epochs, step + 1, len(traindataload), loss.item(), end_time_step - start_time_step
                    )
                )
            with torch.no_grad():
                valdataloader_iter = iter(valdataload)
                # for step, (img, label) in enumerate(valdataload):
                for step in range(len(valdataload)):
                    #             sub_batch = 0
                    #             for img in batch:
                    start_time_step = time.time()
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
                    end_time_step = time.time()
                    #                 sub_batch += 1
                    print(
                        "Epoch [{}/{}], Step [{}/{}], val loss:{:.4f}, time passed:{} ".format(
                            epoch + 1, epochs, step + 1, len(valdataload), loss.item(), end_time_step - start_time_step
                        )
                    )
                    w.write(
                        "Epoch [{}/{}], Step [{}/{}], val loss:{:.4f}, time passed:{} \n".format(
                            epoch + 1, epochs, step + 1, len(valdataload), loss.item(), end_time_step - start_time_step
                        )
                    )

            end_time_epoch = time.time()
            # print('After Epoch [{}/{}], average training loss:{:.4f}, average val loss:{:.4f}'.format(epoch+1,epochs,np.mean(train_loss),np.mean(val_loss)))
            print(
                "After Epoch [{}/{}], average training loss:{:.4f}, average training accuracy {:.4f}, average val loss:{:.4f}, average val {:.4f}, time passed:{}".format(
                    epoch + 1, epochs, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc), end_time_epoch - start_time_epoch
                )
            )

            # w.write('After Epoch [{}/{}], average training loss:{:.4f}, average val loss:{:.4f} \n'.format(epoch+1,epochs,np.mean(train_loss),np.mean(val_loss)))
            w.write(
                " After Epoch [{}/{}], average training loss:{:.4f}, average training accuracy {:.4f}, average val loss:{:.4f}, average val {:.4f}, time passed:{} \n".format(
                    epoch + 1, epochs, np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc), end_time_epoch - start_time_epoch
                )
            )

            current_loss = np.mean(val_loss)
            if best_loss == 0:
                best_loss = current_loss
                print("FIRST EPOCH!!")
                w.write("FIRST EPOCH!! \n")
                torch.save(net.state_dict(), "./models/model_" + str(model_name) + ".pt")
            elif current_loss < best_loss:
                print("GOOD EPOCH!!")
                w.write("GOOD EPOCH!! \n")
                best_loss = current_loss
                torch.save(net.state_dict(), "./models/model_" + str(model_name) + ".pt")
                counter = 0
            else:
                print("BAD EPOCH!!!")
                w.write("BAD EPOCH!! \n")
                counter += 1
                if counter >= patience:
                    print("MODEL STARTS TO OVERFIT!!!")
                    w.write("MODEL STARTS TO OVERFIT!!! \n")
                    break
            train_loss_final.append(np.mean(train_loss))
            train_acc_final.append(np.mean(train_acc))
            val_loss_final.append(np.mean(val_loss))
            val_acc_final.append(np.mean(val_acc))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 20))
        ax1.plot(train_loss_final, label="Training loss")
        ax1.plot(val_loss_final, label="Validation loss")
        ax1.legend(loc="upper right")
        ax2.plot(train_acc_final, label="Training accuracy")
        ax2.plot(val_acc_final, label="Validation accuracy")
        ax2.legend(loc="lower right")
        ax2.set_ylim([0, 100])

        # plt.figure(figsize=(30,20))
        # plt.plot(train_loss_final,label = 'Training loss')
        # plt.plot(val_loss_final,label='Validation loss')
        # plt.legend(loc = 'upper right')
        plt.savefig("./reports/models/learning_curves_" + str(model_name) + ".png")


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
