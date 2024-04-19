import pandas as pd
import torch
from nd_dataloader import Dataloader as nd_Dataloader
from torch.utils.data import DataLoader

# from torchsummary import summary
from torch_trandsforms import Compose, RandomFlip, RandomRotate, Resize

from src.data.make_dataset import CustomDataset

# from src.models.generate_model import main
from src.models.train_model import train3D

# from src.features.utils import evaluate


def get_augmentation(patch_size, train=False):
    if train:
        return Compose(
            [
                RandomRotate((-15, 15), (0, 0), (0, 0), p=0.5),
                # RandomCropFromBorders(crop_value=0.1, p=0.5),
                # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
                Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
                RandomFlip(0, p=0.5),
                RandomFlip(1, p=0.5),
                RandomFlip(2, p=0.5),
                # RandomRotate90((1, 2), p=0.5),
                # GaussianNoise(var_limit=(0, 5), p=0.2),
                # RandomGamma(gamma_limit=(80, 120), p=0.2),
            ],
            p=1.0,
        )
    else:
        return Compose(
            [
                # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
                # RandomCropFromBorders(crop_value=0.1, p=0.5),
                # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
                Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
                # Flip(0, p=0.5),
                # Flip(1, p=0.5),
                # Flip(2, p=0.5),
                # RandomRotate90((1, 2), p=0.5),
                # GaussianNoise(var_limit=(0, 5), p=0.2),
                # RandomGamma(gamma_limit=(80, 120), p=0.2),
            ],
            p=1.0,
        )


if __name__ == "__main__":
    EPOCHS = 10
    CHANNEL = 1
    DIM = 3
    BATCH_SIZE = 2
    NUM_WORKERS = 10
    STEPS_PER_EPOCH = 100
    DEVICE = torch.device("cuda:0")
    assert torch.cuda.is_available(), "CUDA not available"
    TENSOR_DTYPE = torch.float16
    TARGET_DTYPE = torch.long
    HEIGHT = 500
    WIDTH = 500
    DEPTH = 250
    DATASET_LENGTH = 32

    batch_size = (BATCH_SIZE,)
    num_workers = (NUM_WORKERS,)
    device = (DEVICE,)
    height = (64,)
    width = (64,)
    depth = (64,)
    length = (2**12,)
    memory_format = (torch.contiguous_format,)
    tensor_dtype = (TENSOR_DTYPE,)
    target_dtype = (TARGET_DTYPE,)

    # LOADING IN THE SPLITS

    train_df = pd.read_csv("./data/train_split_filtered.csv")
    val_df = pd.read_csv("./data/val_split_filtered.csv")
    test_df = pd.read_csv("./data/test_split_filtered.csv")

    ################################################################################################################################################################################################################################

    # 2D DATASET

    # traindataset = CustomDataset2D(train_df,3)
    # valdataset = CustomDataset2D(val_df,3)
    # testdataset = CustomDataset2D(test_df,3)

    # traindataset = CustomDataset2Dslices(train_df,4,2)
    # traindataloader = DataLoader(traindataset,batch_size=5,shuffle=True)
    # valdataset = CustomDataset2Dslices(val_df,4,2)
    # valdataloader = DataLoader(valdataset,batch_size=2)

    # testdataset = CustomDataset2Dslices(test_df,8,2)
    # testdataloader = DataLoader(testdataset,batch_size=1)

    ################################################################################################################################################################################################################################
    # 3D DATASET

    transform_train = get_augmentation((250, 500, 500), True)
    transform_val = get_augmentation((250, 500, 500))

    traindataset = CustomDataset(train_df, transform=transform_train)
    traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True)
    traindataloader = nd_Dataloader(
        traindataset,
        batch_size=batch_size,
        num_workers=num_workers,
        memory_format=memory_format,
        use_prefetcher=True,
        mean=traindataset.mean,
        std=traindataset.std,
        dim=DIM,
        is_training=True,
        pin_memmory=False,
        tensor_dtype=torch.float16,
        target_dtype=torch.long,
        batch_norm=True,
    )
    valdataset = CustomDataset(val_df, transform=transform_val)
    valdataloader = DataLoader(valdataset, batch_size=1)
    valdataloader = nd_Dataloader(
        valdataset,
        batch_size=batch_size,
        num_workers=num_workers,
        memory_format=memory_format,
        use_prefetcher=True,
        mean=valdataset.mean,
        std=valdataset.std,
        dim=DIM,
        is_training=True,
        pin_memmory=False,
        tensor_dtype=torch.float16,
        target_dtype=torch.long,
        batch_norm=True,
    )
    testdataset = CustomDataset(test_df, transform=transform_val)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=True)
    dataloader = nd_Dataloader(
        testdataset,
        batch_size=batch_size,
        num_workers=num_workers,
        memory_format=memory_format,
        use_prefetcher=True,
        mean=testdataset.mean,
        std=testdataset.std,
        dim=DIM,
        is_training=True,
        pin_memmory=False,
        tensor_dtype=torch.float16,
        target_dtype=torch.long,
        batch_norm=True,
    )
    ################################################################################################################################################################################################################################
    # 3D training

    train3D(traindataloader, valdataloader, "model_test_v6", 150, 40, 0.0001)

    # evaluate(testdataset,'./results/models/model_test_v6.pt')
    # evaluate2D(testdataset,'./results/models/model_resnet2D_4slices_fulltraining.pt')

    ################################################################################################################################################################################################################################
    # 2D training
    # print(f"Training dataset size: {len(traindataset)}, validation dataset size {len(valdataset)}")
    # train_multiview(traindataloader, valdataloader, "multiview_4slices_no_aug_CTRL_100_v1", 500, 500, 0.0001)

    # evaluate2D(testdataset,'./results/models/model_multiview_8slices_no_aug_v2.pt')

    # img, tar = next(iter(traindataloader))
    # print(img.shape)
    # print(tar)
    # # net = SeparateChannelsResNet18()

    # i = io.imread('./data/20230502/100/2F_8slices.tif')
    # print(i.shape)
