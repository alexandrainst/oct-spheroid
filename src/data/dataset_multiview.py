import os

import mat73
import numpy as np
import scipy
import torch
from skimage.transform import resize, rotate
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

DEBUG = False


class MultiviewDataset(Dataset):
    def __init__(
        self,
        df,
        n_slices=4,
        n_classes=2,
        labels_dict={"CTRL": 0, "25": 1, "100": 2},
        transform=None,
        return_filename=False,
        store_pp=True,
        store_pp_path="/dsv/cancer",
        norm=True,
    ):
        self.return_filename = return_filename
        self.labels_dict = labels_dict
        self.store_pp = store_pp  # boolean to store preprocessed images
        self.store_pp_path = store_pp_path  # path to store preprocessed images
        self.mean = [12.4361, 12.4864, 12.4278, 12.3149]
        self.std = [10.7506, 10.7760, 9.8911, 9.9736]
        self.norm = transforms.Normalize(mean=self.mean, std=self.std)
        self.transform = transform
        if n_classes == 2:
            k = list(labels_dict.keys())
            self.df = df[(df["labels"] == k[0]) | (df["labels"] == k[1])]
        else:
            self.df = df

        self.n_slices = n_slices
        self.norm_bool = norm
        ending = "_" + str(self.n_slices) + "slices.npy"

        self.labels = []
        self.filepaths_slices = []
        self.filepaths = []
        for i in tqdm(range(len(self.df)), desc="Preparing dataset", position=0, leave=True, total=len(self.df)):
            filepath = self.df.iloc[i, -2]
            self.filepaths.append(filepath)
            filename_slice = filepath.replace(".mat", ending)
            if self.store_pp:
                filename_slice = os.path.join(self.store_pp_path, *filename_slice.split("/")[-3:])
            self.filepaths_slices.append(filename_slice)
            self.labels.append(self.df.iloc[i, -1])

        self.labels = [self.labels_dict[i] for i in self.labels]

    def __getitem__(self, index):
        filename = self.filepaths[index]
        filename_slice = self.filepaths_slices[index]
        label = self.labels[index]
        if os.path.exists(filename_slice):
            sample = np.load(filename_slice)
        else:
            sample = self.get_slices(filename)
            assert sample.shape[0] == self.n_slices, f"Expected {self.n_slices}, got {sample.shape[0]}"
            if sample.shape != (self.n_slices, 250, 1000):
                print(f"Expected shape {(self.n_slices, 250, 1000)}, got {sample.shape} - resizing")
                sample = resize(sample, (self.n_slices, 250, 1000))
            if self.store_pp:
                os.makedirs(os.path.dirname(filename_slice), exist_ok=True)
                np.save(filename_slice, sample)

        assert sample.shape == (self.n_slices, 250, 1000), f"Expected shape {(self.n_slices, 250, 1000)}, got {sample.shape}"

        if DEBUG:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.imshow(sample[i, :, :])
            plt.savefig(f"getitem_idx_{index}.png")
            plt.close()

        sample_torch = torch.from_numpy(sample)
        if self.transform is not None:
            sample_torch = self.transform(sample_torch)

        assert sample_torch.shape == (self.n_slices, 250, 1000), f"Expected shape {(self.n_slices, 250, 1000)}, got {sample_torch.shape}"
        if DEBUG:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            for i in range(4):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.imshow(sample_torch[i, :, :])
            plt.savefig(f"getitem_after_norm_idx_{index}.png")
            plt.close()

        if self.return_filename:
            return sample_torch, label, filename_slice
        else:
            return sample_torch, label

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
