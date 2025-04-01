import os

import mat73
import numpy as np
import scipy
from skimage import io
from skimage.transform import resize, rotate
from torch.utils.data import Dataset
from torchvision import ops, transforms
from tqdm import tqdm


class MultiviewDataset(Dataset):
    def __init__(self, df, n_slices=4, n_classes=2, labels_dict={"CTRL": 0, "25": 1, "100": 2}, transform=None, save=False, return_filename=False):
        self.return_filename = return_filename
        self.labels_dict = labels_dict
        self.mean = [9.713811]
        self.std = [10.071295]
        if n_classes == 2:
            k = list(labels_dict.keys())
            self.df = df[(df["labels"] == k[0]) | (df["labels"] == k[1])]
        else:
            self.df = df

        self.n_slices = n_slices

        ending = "_" + str(self.n_slices) + "slices.tif"

        self.images = []
        self.labels = []
        self.filename_slices = []
        self.filename = []
        for i in tqdm(range(len(self.df)), desc="Preparing dataset", position=0, leave=True, total=len(self.df)):
            filename = self.df.iloc[i, -2]
            self.filename.append(filename)
            filename_slice = filename.replace(".mat", ending)
            self.filename_slices.append(filename_slice)
            if os.path.exists(filename_slice):
                self.images.append(io.imread(filename_slice))
            else:
                self.images.append(self.get_slices(filename))
            self.labels.append(self.df.iloc[i, -1])
        self.images = [resize(i, (self.n_slices, 250, 1000)) for i in tqdm(self.images, desc="Resizing images", position=0, leave=True, total=len(self.images))]
        self.labels = [self.labels_dict[i] for i in self.labels]
        if save:
            for i in tqdm(range(len(self.df)), desc="Saving images", position=0, leave=True, total=len(self.df)):
                if not os.path.exists(self.filename_slices[i]):
                    print(self.filename_slices[i])
                    io.imsave(self.filename_slices[i], self.images[i])

    def __getitem__(self, index):
        sample = self.images[index]
        label = self.labels[index]
        if sample.shape != (self.n_slices, 250, 1000):
            sample = resize(sample, (self.n_slices, 250, 1000))
        transform_backbone = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[9.713811], std=[10.071295]), ops.Permute([1, 2, 0])])

        sample = transform_backbone(sample)
        if self.return_filename:
            filename = self.filename[index]
            return sample, label, filename
        else:
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
        vol_thresh = (vol > 0.9).astype(float)
        median_pos = [int(np.median(i)) for i in np.where(vol_thresh == 1)]

        sliced_input = np.zeros((self.n_slices, vol.shape[0], vol.shape[2]))
        sliced_input_2drot = np.zeros((vol.shape[0], vol.shape[1], vol.shape[2]))
        for i in range(int(self.n_slices / 2)):  # , desc="Rotating volume", position=0, leave=True, total=int(self.n_slices/2)):
            if i == 0:
                sliced_input[0] = vol[:, median_pos[1], :]
                sliced_input[1] = vol[:, :, median_pos[2]]
            else:
                idx = i * 2
                for i in range(vol.shape[0]):
                    sliced_input_2drot[i, :, :] = rotate(vol[i, :, :], 45)
                sliced_input[idx] = sliced_input_2drot[:, median_pos[1], :]
                sliced_input[idx + 1] = sliced_input_2drot[:, :, median_pos[2]]

        return np.array(sliced_input)

    def __len__(self):
        return len(self.df)
