import random

import mat73
import numpy as np
import scipy
import torch
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset
from torch_trandsforms import Compose, RandomRotate
from torchvision import ops, transforms
from tqdm import tqdm

random.seed(45)


class CustomDataset2D(Dataset):
    def __init__(self, df, n_classes=2, transform=None):
        if n_classes == 2:
            self.labels_dict = {"CTRL": 2, "25": 0, "100": 1}
            self.df = df[(df["labels"] == "25") | (df["labels"] == "100")]
        else:
            self.labels_dict = {"CTRL": 0, "25": 1, "100": 2}
            self.df = df
        # self.df = df[(df['labels'] == 'CTRL') | (df['labels'] == '100')]
        # self.df = df[(df['labels'] == '25')]

        self.transform = Compose(
            [
                RandomRotate([90, 90, 90], sample_mode="nearest", p=0.9),
            ]
        )

        self.images = []
        self.labels = []
        for i in tqdm(range(len(self.df)), desc="Preparing dataset", position=0, leave=True, total=len(self.df)):
            self.images.append(self.get_slices(self.df.iloc[i, 0]))
            self.labels.append(self.df.iloc[i, 1])
        self.images = [resize(i, (8, 250, 1000)) for i in self.images]
        self.labels = [self.labels_dict[i] for i in self.labels]
        for i in tqdm(range(len(self.df)), desc="Saving images", position=0, leave=True, total=len(self.df)):
            io.imsave(self.df.iloc[i, 0].replace(".mat", "_8slices.tif"), self.images[i])

    def __getitem__(self, index):
        # row = self.df.iloc[index,0]
        sample = self.images[index]
        label = self.labels[index]

        transform_backbone = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[9.713811], std=[10.071295]), ops.Permute([1, 2, 0])])
        sample = transform_backbone(sample)

        return sample, label

    def load_matfile(self, path):
        try:
            img = scipy.io.loadmat(path)
            # print('Loaded with scipy')
        except NotImplementedError:
            try:
                img = mat73.loadmat(path)
                # print('Loaded with mat73')
            except NotImplementedError:
                print("Cannot load file!!")

        return img["tdData"]

    def get_slices(self, path):
        vol = self.load_matfile(path)
        # vol = normalise_to_positive(vol)
        vol_thresh = (vol > 0.9).astype(float)
        median_pos = [int(np.median(i)) for i in np.where(vol_thresh == 1)]
        # sample = self.transform(**{'image':vol})

        sliced_input = np.zeros((8, vol.shape[0], vol.shape[2]))
        for i in tqdm(range(4), desc="Rotating volume", position=0, leave=True, total=4):
            if i == 0:
                sliced_input[0] = vol[:, median_pos[1], :]
                sliced_input[1] = vol[:, :, median_pos[2]]
            else:
                idx = i * 2
                sample = self.transform(**{"image": vol})
                sample = sample["image"]
                sliced_input[idx] = sample[:, median_pos[1], :]
                sliced_input[idx + 1] = sample[:, :, median_pos[2]]

        # sliced_input[2] = sample[:,median_pos[1],:]
        # sliced_input[3] = sample[:,:,median_pos[2]]
        return sliced_input

    def __len__(self):
        return len(self.df)


class CustomDataset2Dslices(Dataset):
    def __init__(self, df, n_slices, n_classes=2, transform=None):
        self.n_slices = str(n_slices)
        if n_classes == 2:
            self.labels_dict = {"CTRL": 0, "25": 2, "100": 1}
            self.df = df[(df["labels"] == "CTRL") | (df["labels"] == "100")]
        else:
            self.labels_dict = {"CTRL": 0, "25": 1, "100": 2}
            self.df = df
        # self.df = df

        self.transform = Compose(
            [
                RandomRotate([90, 90, 90], sample_mode="nearest", p=0.9),
            ]
        )

        ending = "_" + self.n_slices + "slices.tif"
        self.images = []
        self.labels = []
        for i in tqdm(range(len(self.df)), desc="Preparing dataset", position=0, leave=True, total=len(self.df)):
            filename = self.df.iloc[i, 0].replace(".mat", ending)
            self.images.append(io.imread(filename))
            self.labels.append(self.df.iloc[i, 1])
        self.labels = [self.labels_dict[i] for i in self.labels]

    def __getitem__(self, index):
        # row = self.df.iloc[index,0]
        sample = self.images[index]
        label = self.labels[index]

        transform_backbone = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[9.713811], std=[10.071295]),
                # ops.Permute([1,2,0])
            ]
        )
        sample = transform_backbone(sample)

        return sample, label

    def load_matfile(self, path):
        try:
            img = scipy.io.loadmat(path)
            # print('Loaded with scipy')
        except NotImplementedError:
            try:
                img = mat73.loadmat(path)
                # print('Loaded with mat73')
            except NotImplementedError:
                print("Cannot load file!!")

        return img["tdData"]

    def get_slices(self, path):
        vol = self.load_matfile(path)
        # vol = normalise_to_positive(vol)
        vol_thresh = (vol > 0.9).astype(float)
        median_pos = [int(np.median(i)) for i in np.where(vol_thresh == 1)]
        sample = self.transform(**{"image": vol})
        sample = sample["image"]
        sliced_input = np.zeros((4, vol.shape[0], vol.shape[2]))
        sliced_input[0] = vol[:, median_pos[1], :]
        sliced_input[1] = vol[:, :, median_pos[2]]
        sliced_input[2] = sample[:, median_pos[1], :]
        sliced_input[3] = sample[:, :, median_pos[2]]
        return sliced_input

    def __len__(self):
        return len(self.df)


class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.labels_dict = {"CTRL": 0, "25": 2, "100": 1}
        # self.df = df
        self.df = df[(df["labels"] == "CTRL") | (df["labels"] == "100")]

        # self.folder = folder
        # self.files = glob.glob(os.path.join(folder,'*/*'))
        self.transform = transform
        # self.samples = [(i,labels_dict[str(i.split('/')[-2])]) for i in self.files]

    def __getitem__(self, index):
        # row = self.df.iloc[index,0]
        img_name, label = self.df.iloc[index, 0], self.df.iloc[index, 1]
        label = self.labels_dict[label]
        # path, target = self.samples[index]
        # img_name = img_name.replace('mat','tif')
        # print(f'Image name:{img_name}')
        sample = self.load_matfile(img_name)
        # sample = io.imread(img_name)
        # sample = normalise_to_positive(sample)
        # cv2.imshow('before',sample[0,:,:])
        # plt.imshow(sample[150,:,:])
        # plt.savefig('before.pdf')
        # plt.show()
        # sample = resize(sample,(50,100,100))

        # plt.imshow(sample[150,:,:])
        # plt.savefig('after.pdf')

        # cv2.imshow('after',sample[0,:,:])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(sample[0,:,:])
        # plt.show()
        # sample = sample[:300,:650,:650]
        if self.transform is not None:
            # data = {'image':sample}
            sample = self.transform(**{"image": sample})
            sample = sample["image"]
        transform_backbone = transforms.Compose(
            [
                transforms.ToTensor(),
                ops.Permute([1, 2, 0]),
            ]
        )
        sample = transform_backbone(sample)
        sample = torch.unsqueeze(sample, 0)

        return sample, label

    def load_matfile(self, path):
        try:
            img = scipy.io.loadmat(path)
            # print('Loaded with scipy')
        except NotImplementedError:
            try:
                img = mat73.loadmat(path)
                # print('Loaded with mat73')
            except NotImplementedError:
                print("Cannot load file!!")

        return img["tdData"]

    def __len__(self):
        return len(self.df)
