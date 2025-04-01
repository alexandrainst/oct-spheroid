import os

import mat73
import numpy as np
import scipy
import torch
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import ops, transforms
from tqdm import tqdm


class CancerSphere3DDataset(Dataset):
    """
    Dataset class for the CancerSphere dataset that preprocesses the images and stores them as numpy arrays.
    This still allows for the use of transforms, but decreases loading time.
    """

    def __init__(self, df, n_classes, reshape_size, transform=None, labels_dict={"CTRL": 0, "25": 1, "100": 2}, store_pp=True, store_pp_path="/dsv/cancer"):
        self.labels_dict = labels_dict
        key_list = list(labels_dict.keys())
        if n_classes == 2:
            self.df = df[(df["labels"] == key_list[0]) | (df["labels"] == key_list[1])]
        else:
            self.df = df

        self.transform = transform
        self.reshape_size = reshape_size
        self.resolution = ("_").join((str(reshape_size[0]), str(reshape_size[1]), str(reshape_size[2])))
        self.mean = [9.713811]
        self.std = [10.071295]

        self.store_pp = store_pp  # boolean to store preprocessed images
        self.store_pp_path = store_pp_path  # path to store preprocessed images
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                ops.Permute([1, 2, 0]),
            ]
        )

    def load_matfile(self, path):
        try:
            img = scipy.io.loadmat(path)
            # print('Loaded with scipy')
        except NotImplementedError:
            try:
                img = mat73.loadmat(path)
                # print('Loaded with mat73')
            except TypeError:
                print("Cannot load file!!")
                print(path)

        return img["tdData"]

    def preprocess(self):
        """
        Iterates over the dataset and preprocesses the images in order to save them as numpy arrays
        """
        for i in tqdm(range(len(self.df)), desc="Preprocessing images", position=0, leave=True, total=len(self.df)):
            img_name = self.df.iloc[i, 1]
            new_img_name = img_name.split(".m")[0] + "_" + str(self.resolution) + ".npy"
            # replace "data" with "data_tmp"
            new_img_name = new_img_name.replace("data", "data/data_tmp")
            if os.path.exists(new_img_name):
                continue
            sample = self.load_matfile(img_name)
            sample = resize(sample, self.reshape_size)
            sample = sample.astype(np.float16)
            os.makedirs(os.path.dirname(new_img_name), exist_ok=True)
            np.save(new_img_name, sample)

    def __getitem__(self, index):
        # row = self.df.iloc[index,0]
        img_path, label = self.df.iloc[index, -2], self.df.iloc[index, -1]
        label = self.labels_dict[label]
        pp_img_path = os.path.join(self.store_pp_path, *img_path.split("/")[-3:])
        pp_img_path = os.path.splitext(pp_img_path)[0] + "_" + str(self.resolution) + ".npy"

        try:
            sample = np.load(pp_img_path)
            # print(new_img_name)
            assert sample.dtype == np.float16, f"{self.__class__.__name__} Expected float16, got {sample.dtype}"
        except FileNotFoundError:
            print(self.__class__.__name__, "File not found, loading from mat file and saving as npy to pre-process-path")
            sample = self.load_matfile(img_path)
            sample = resize(sample, self.reshape_size)
            # cast to float16
            sample = sample.astype(np.float16)
            assert sample.dtype == np.float16, f"{self.__class__.__name__} Expected float16, got {sample.dtype}"
            os.makedirs(os.path.dirname(pp_img_path), exist_ok=True)
            np.save(pp_img_path, sample)
        except Exception as e:
            print(e)
            raise e

        if self.transform is not None:
            raise NotImplementedError(f"{self.__class__.__name__} Before applying transforms, we need to check if they are compatible with float16")
            sample = self.transform(**{"image": sample})
            sample = sample["image"]

        sample = self.to_tensor(sample)
        sample = torch.unsqueeze(sample, 0)  # TODO: check if we could move that in the collate function and pre-allcate the memory there
        assert sample.dtype == torch.float16, f"{self.__class__.__name__} Expected float16, got {sample.dtype}"

        label = torch.tensor(label, dtype=torch.long)
        return sample, label
