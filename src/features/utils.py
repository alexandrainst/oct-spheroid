import mat73
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.generate_model import generate_model
from src.models.multiview import SeparateChannelsResNet18


def normalise_to_positive(vol):
    vol_norm = vol.astype("float32")
    # calculate global mean and standard deviation
    mean, std = vol_norm.mean(), vol_norm.std()
    # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    vol_norm = (vol_norm - mean) / std
    # clip pixel values to [-1,1]
    vol_norm = np.clip(vol_norm, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    vol_norm = (vol_norm + 1.0) / 2.0
    # confirm it had the desired effect
    mean, std = vol_norm.mean(), vol_norm.std()
    # print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # print('Min: %.3f, Max: %.3f' % (vol_norm.min(), vol_norm.max()))
    return vol_norm


class SphereHandler:
    def __init__(self, path, thresh, norm=False, vis=False, debug=False):
        self.path = path
        self.thresh = thresh
        self.norm = norm
        self.vis = vis
        self.debug = debug
        self.filename = self.path.split("/")[-1].split(".")[0]

    def remove_specles(self, img, plot=False, count=0):
        test_thresh = (img > self.thresh).astype(float)
        # test_fil = cv.morphologyEx(test_thresh,
        #                               cv.MORPH_TOPHAT,
        #                               kernel)
        test_fil = skimage.measure.label(test_thresh, connectivity=2)
        uni, counts = np.unique(test_fil, return_counts=True)
        idx = [i for i in range(len(counts)) if counts[i] > 10000]
        good_labels = [uni[i] for i in idx]
        bad_labels = [i for i in uni if i not in good_labels]
        for i in bad_labels:
            test_fil[test_fil == i] = 0

        test_thresh_fil = test_thresh.copy()
        test_thresh_fil[test_fil == 0] = 0

        if plot:
            # plt.imshow(test_thresh)

            plt.clf()
            plt.figure(figsize=(30, 30))
            plt.subplot(411)
            plt.imshow(img, cmap="gray")
            plt.subplot(412)
            plt.imshow(test_thresh, cmap="gray")
            plt.subplot(413)
            plt.imshow(test_fil)
            plt.subplot(414)
            plt.imshow(test_thresh_fil, cmap="gray")
            # plt.show()
            plt.savefig(f"thresh_test_{count}_{self.filename}.png")
        return test_thresh_fil

    def find_width(self, img, plot=False, debug=False, count=0):
        if debug:
            test_thresh = self.remove_specles(img, True, count)
        else:
            test_thresh = self.remove_specles(img)
        rays = 0
        row = 0
        end_f = 0
        begin_f = 0
        for i in range(test_thresh.shape[1]):
            begin = 0
            end = 0
            x = test_thresh[i, :]
            for j in range(len(x)):
                if x[j] == 1:
                    begin = j
            for k in range(len(x) - 1, -1, -1):
                if x[k] == 1:
                    end = k
            length = end - begin
            #         rays.append(-length)
            if -length > rays:
                rays = -length
                row = i
                end_f = end
                begin_f = begin
        #     print(rays)
        if plot:
            plt.figure(figsize=(10, 10))
            plt.subplot(121)
            plt.imshow(img, cmap="gray")
            plt.plot(
                [
                    i
                    for i in range(
                        end_f,
                        begin_f,
                    )
                ],
                [
                    row
                    for i in range(
                        end_f,
                        begin_f,
                    )
                ],
            )
            plt.title("Original Image")
            plt.axis("off")
            plt.subplot(122)
            plt.imshow(test_thresh, cmap="gray")
            plt.plot(
                [
                    i
                    for i in range(
                        end_f,
                        begin_f,
                    )
                ],
                [
                    row
                    for i in range(
                        end_f,
                        begin_f,
                    )
                ],
            )
            plt.title("Filtered Image")
            plt.axis("off")
            plt.show()
        return rays

    def plot_sphere(self, volume):
        all_widths = []
        for i in tqdm(volume, position=0, leave=True):
            width = self.find_width(i, self.thresh)
            all_widths.append(width)
        b, e, s1 = self.find_sphere_on_graph(all_widths)
        plt.imshow(volume[:, 500, :], cmap="gray")
        plt.plot([i for i in range(1000)], [int(b) for i in range(1000)])
        plt.plot([i for i in range(1000)], [int(e) for i in range(1000)])
        plt.show()

        return

    def find_sphere_on_graph(self, width_list, plot=False, debug=True):
        #     b = []
        #     e = []
        b = 0
        e = 0
        for i in range(len(width_list)):
            if i < len(width_list) - 150:
                if width_list[i] == 0:
                    b = i
            else:
                if np.mean(width_list[i : i + 10]) > width_list[i] + 200:
                    # print('end sample')
                    #                     l = [-3,-2,-1,0,1,2,3]
                    idx = np.argmin(width_list[i : i + 5])
                    e = i + idx
                    break

        if debug:
            plt.figure(figsize=(10, 5))
            plt.plot(width_list)
            plt.scatter(b, width_list[b], color="red")
            plt.scatter(e, width_list[e], color="green")
            plt.savefig(f"widths_{self.filename}.png")

        if plot:
            plt.figure(figsize=(30, 20))
            # plt.subplot(311); plt.plot(width_list); plt.scatter( begin, width_list[begin] , color='red'); plt.scatter( end, width_list[end], color='green'); plt.title('Width across volume 1 smoothed')
            plt.subplot(311)
            plt.plot(
                width_list
            )  # ; plt.scatter( begin, width_list[begin] , color='red'); plt.scatter( end, width_list[end], color='green'); plt.title('Width across volume 1 smoothed')
            plt.subplot(312)
            plt.plot(width_list[b:e])
        return b, e, width_list[b:e]

    def find_sphere_on_graph_no_histogel(self, width_list, plot=False, debug=True):
        b = []
        e = []
        for i in range(len(width_list[30:])):
            if i == len(width_list[30:]) - 4:
                break
            else:
                if width_list[30:][i] - width_list[30:][i + 1] > 200:
                    b.append(i + 1)
                elif np.mean(width_list[30:][i : i + 5]) > width_list[30:][i] + 100:
                    e.append(i)

        print(b)
        print(e)

        test_b = [i + 30 for i in b]
        test_e = [i + 30 for i in e]
        test_c = []
        for i in range(len(test_e)):
            if i == len(test_e) - 1:
                test_c.append(test_e[i])
                break
            elif test_e[i + 1] - test_e[i] > 1:
                test_c.append(test_e[i])
        #     test_e = [test_e[i] + 30 for i in range(len(test_e)) if i > 0 and i < len(test_e) and test_e[i+1] - test_e[i] > 1 ]

        #     test_e = [i for i in test_e if (i+1) - i > 1]
        print(test_c)
        diff = 1000000
        begin = [i for i in test_b if i < len(width_list) / 2]
        begin = max(begin)
        for i in test_c:
            if i - begin < 0:
                continue
            elif i - begin < diff:
                diff = i - begin
                end = i
        if debug:
            plt.figure(figsize=(10, 5))
            plt.plot(width_list)
            plt.scatter(test_b, [width_list[i] for i in test_b], color="red")
            plt.scatter(test_c, [width_list[i] for i in test_c], color="green")

        if plot:
            plt.figure(figsize=(30, 20))
            # plt.subplot(311); plt.plot(width_list); plt.scatter( begin, width_list[begin] , color='red'); plt.scatter( end, width_list[end], color='green'); plt.title('Width across volume 1 smoothed')
            plt.subplot(311)
            plt.plot(
                width_list
            )  # ; plt.scatter( begin, width_list[begin] , color='red'); plt.scatter( end, width_list[end], color='green'); plt.title('Width across volume 1 smoothed')
            plt.subplot(312)
            plt.plot(width_list[begin:end])
        return begin, end, width_list[begin:end]

    def isolate_sphere(self, vis=True):
        v = self.load_matfile()
        if self.norm:
            v = normalise_to_positive(v)
        # m = np.mean([np.mean(j) for j in v])
        # if m < 40:
        #     t = 50
        # else:
        #     t = 60
        all_widths = []
        counter = 0
        for i in tqdm(v, position=0, leave=True):
            if counter % 50 == 0:
                width = self.find_width(i, False, True, counter)
            else:
                width = self.find_width(i)
            all_widths.append(width)
            counter += 1

        b, e, s = self.find_sphere_on_graph(all_widths, debug=vis)
        if vis:
            # plt.show()
            plt.clf()
            plt.imshow(v[:, 500, :], cmap="gray")
            plt.plot([i for i in range(1000)], [int(b) for i in range(1000)])
            plt.plot([i for i in range(1000)], [int(e) for i in range(1000)])
            plt.savefig(f"sphere_isolated_{self.filename}.png")

        return v[b:e, :, :]

    def load_matfile(self):
        try:
            img = scipy.io.loadmat(self.path)
            return img["tdData"]
        except NotImplementedError:
            try:
                img = mat73.loadmat(self.path)
                return img["tdData"]
            except NotImplementedError:
                print("Cannot load file!!")


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


def evaluate2D(testdataset, weights):
    testDataLoader = DataLoader(testdataset, batch_size=1)
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

    net = SeparateChannelsResNet18(num_classes=3)
    # net = net.float()
    net.load_state_dict(torch.load(weights))
    net = net.to(device)
    labels = []
    preds = []
    with open("./results/models/evaluation_log_" + str(weights.split("/")[-1].split(".")[0]) + ".txt", "w") as w:
        with torch.no_grad():
            for step, (img, label) in tqdm(enumerate(testDataLoader), desc="Test evaluation", leave=True, position=0, total=len(testDataLoader)):
                filename = testdataset.df.iloc[step, 0].split("/")[-1]
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
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
    plt.figure(figsize=(20, 20))
    cm_display.plot()
    plt.show()
    plt.savefig("evaluation_confusion_matrix_" + str(weights.split("/")[-1].split(".")[0]) + ".png")


def evaluate(testdataset, weights):
    testDataLoader = DataLoader(testdataset, batch_size=1, shuffle=True)
    acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     device = torch.device('cpu')
    net = generate_model("resnet", 10, 2, 1, 128)  # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Modify the last layer of the model to match the number of texture classes in DTD
    num_texture_classes = 2
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, num_texture_classes)

    # Replace the original first layer with the new one
    net.load_state_dict(torch.load(weights))
    net = net.to(device)
    labels = []
    preds = []
    with open("./results/models/evaluation_log_" + str(weights.split("/")[-1].split(".")[0]) + ".txt", "w") as w:
        with torch.no_grad():
            for step, (img, label) in tqdm(enumerate(testDataLoader), desc="Test evaluation", leave=True, position=0, total=len(testDataLoader)):
                filename = testdataset.df.iloc[step, 0].split("/")[-1]
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
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    plt.figure(figsize=(20, 20))
    cm_display.plot()
    plt.show()
    plt.savefig("evaluation_confusion_matrix_" + str(weights.split("/")[-1].split(".")[0]) + ".png")
