import os
import random

import mat73
import numpy as np
import scipy
import torch
from skimage import io
from skimage.transform import resize, rotate
from torch.utils.data import Dataset
from torch_trandsforms import Compose, RandomRotate  # , Normalize
from torchvision import ops, transforms
from tqdm import tqdm

# from volumentations import Rotate

random.seed(45)


class CustomDataset2D(Dataset):
    def __init__(self, df, n_slices=4, n_classes=2, labels_dict={"CTRL": 0, "25": 1, "100": 2}, transform=None, save=False, return_filename=False):
        self.return_filename = return_filename
        self.labels_dict = labels_dict
        self.mean = [9.713811]
        self.std = [10.071295]
        if n_classes == 2:
            # self.labels_dict = {"CTRL": 2, "25": 0, "100": 1}
            k = list(labels_dict.keys())
            # print(k)
            self.df = df[(df["labels"] == k[0]) | (df["labels"] == k[1])]
        else:
            # self.labels_dict = {"CTRL": 0, "25": 1, "100": 2}
            self.df = df
        # self.df = df[(df['labels'] == 'CTRL') | (df['labels'] == '100')]
        # self.df = df[(df['labels'] == '25')]

        self.n_slices = n_slices

        # self.rotator = Compose([
        #     # Normalize(mean = [9.713811],std=[10.071295]),
        #     RandomRotate([180,0,0], sample_mode="nearest", p=1)
        # ])
        # self.transform = Compose(
        #     [
        #         Rotate((0, 0), (0, 0), (45, 45), p=1),
        #     ],
        #     p=1.0,
        # )

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
        # row = self.df.iloc[index,0]
        sample = self.images[index]
        label = self.labels[index]
        if sample.shape != (self.n_slices, 250, 1000):
            sample = resize(sample, (self.n_slices, 250, 1000))
        transform_backbone = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[9.713811], std=[10.071295]), ops.Permute([1, 2, 0])])
        # transform_backbone = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[9.713811], std=[10.071295])])
        # transform_backbone = transforms.Compose([transforms.ToTensor(), ops.Permute([1, 2, 0])])

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
        # vol = normalise_to_positive(vol)
        vol_thresh = (vol > 0.9).astype(float)
        median_pos = [int(np.median(i)) for i in np.where(vol_thresh == 1)]
        # sample = self.transform(**{'image':vol})

        sliced_input = np.zeros((self.n_slices, vol.shape[0], vol.shape[2]))
        sliced_input_2drot = np.zeros((vol.shape[0], vol.shape[1], vol.shape[2]))
        for i in range(int(self.n_slices / 2)):  # , desc="Rotating volume", position=0, leave=True, total=int(self.n_slices/2)):
            if i == 0:
                sliced_input[0] = vol[:, median_pos[1], :]
                sliced_input[1] = vol[:, :, median_pos[2]]
            else:
                idx = i * 2
                # print(f'original vol min {vol.min()} max {vol.max()}')
                # start = time.time()
                # sample = self.transform(**{"image": vol})
                # sample = sample['image']
                # end = time.time()
                # print(f'3D rotation {end - start}')
                # print(f'3D vol min {sample.min()} max {sample.max()}')

                # vol = torch.from_numpy(vol)
                # vol = vol.unsqueeze(1)
                # sample = self.rotator(input=vol)
                # sample = sample["input"].numpy()

                # sliced_input[idx] = sample[:, median_pos[1], :]
                # sliced_input[idx + 1] = sample[:, :, median_pos[2]]

                # start = time.time()
                for i in range(vol.shape[0]):
                    sliced_input_2drot[i, :, :] = rotate(vol[i, :, :], 45)
                sliced_input[idx] = sliced_input_2drot[:, median_pos[1], :]
                sliced_input[idx + 1] = sliced_input_2drot[:, :, median_pos[2]]
                # end = time.time()
                # print(f'2D rotation {end- start}')
                # print(f'2D vol min {sliced_input_2drot.min()} max {sliced_input_2drot.max()}')
                # print(f'difference {(sample - sliced_input_2drot).sum()}')

        # sliced_input[2] = sample[:,median_pos[1],:]
        # sliced_input[3] = sample[:,:,median_pos[2]]
        return np.array(sliced_input)

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
    def __init__(self, df, n_classes, reshape_size, transform=None, labels_dict={"CTRL": 0, "25": 1, "100": 2}):
        self.labels_dict = labels_dict
        key_list = list(labels_dict.keys())
        if n_classes == 2:
            # self.labels_dict = {"CTRL": 0, "25": 1, "100": 2}
            self.df = df[(df["labels"] == key_list[0]) | (df["labels"] == key_list[1])]
        else:
            # self.labels_dict = {"CTRL": 0, "25": 1, "100": 2}
            self.df = df
        # self.df = df
        # self.df = df[(df["labels"] == "CTRL") | (df["labels"] == "100")]

        # self.folder = folder
        # self.files = glob.glob(os.path.join(folder,'*/*'))
        self.transform = transform
        self.reshape_size = reshape_size
        self.resolution = ("_").join((str(reshape_size[0]), str(reshape_size[1]), str(reshape_size[2])))
        self.mean = [9.713811]
        self.std = [10.071295]

        # self.mean = [0.0139984]
        # self.std = [0.1418142]

        # self.samples = [(i,labels_dict[str(i.split('/')[-2])]) for i in self.files]

    def __getitem__(self, index):
        # row = self.df.iloc[index,0]
        img_name, label = self.df.iloc[index, -2], self.df.iloc[index, -1]
        label = self.labels_dict[label]
        # print(img_name)
        # path, target = self.samples[index]
        # img_name = img_name.replace('mat','tif')
        # print(f'Image name:{img_name}')
        new_img_name = img_name.split(".m")[0] + "_" + str(self.resolution)
        if os.path.exists(new_img_name + ".npz"):
            try:
                sample = np.load(new_img_name + ".npz")
                sample = sample["image"]
            except Exception as e:
                print(e)
                print(new_img_name)
                sample = self.load_matfile(img_name)
                sample = resize(sample, self.reshape_size)
                np.savez_compressed(new_img_name + ".npz", image=sample)
        else:
            sample = self.load_matfile(img_name)
            sample = resize(sample, self.reshape_size)

        # sample = io.imread(img_name)
        # sample = normalise_to_positive(sample)
        # cv2.imshow('before',sample[0,:,:])
        # plt.imshow(sample[150,:,:])
        # plt.savefig('before.pdf')
        # plt.show()

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

        label = torch.tensor(label, dtype=torch.long)
        return sample, label, img_name

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

    def __len__(self):
        return len(self.df)


class CancerSphere3DDataset(CustomDataset):
    """
    Dataset class for the CancerSphere dataset that preprocesses the images and stores them as numpy arrays.
    This still allows for the use of transforms, but decreases loading time.
    """

    def __init__(self, df, n_classes, reshape_size, transform=None, labels_dict={"CTRL": 0, "25": 1, "100": 2}, store_pp=True, store_pp_path="/dsv/cancer"):
        super().__init__(df, n_classes, reshape_size, transform, labels_dict)
        self.store_pp = store_pp  # boolean to store preprocessed images
        self.store_pp_path = store_pp_path  # path to store preprocessed images
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                ops.Permute([1, 2, 0]),
            ]
        )

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
