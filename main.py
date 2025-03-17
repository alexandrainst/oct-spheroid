import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from nd_dataloader import Dataloader as nd_Dataloader

# from volumentations import *
from torch.utils.tensorboard import SummaryWriter

from src.data.make_dataset import CancerSphere3DDataset

# from src.models.generate_model import main
from src.models.train_model import train3D


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    EPOCHS = 50
    CHANNEL = 1
    DIM = 3
    BATCH_SIZE = 2
    NUM_WORKERS = 4
    STEPS_PER_EPOCH = 100
    DEVICE = torch.device("cuda:0")
    assert torch.cuda.is_available(), "CUDA not available"
    TENSOR_DTYPE = torch.float16
    TARGET_DTYPE = torch.long
    HEIGHT = 500
    WIDTH = 500
    DEPTH = 250
    DATASET_LENGTH = 32

    batch_size = BATCH_SIZE
    num_workers = NUM_WORKERS
    device = DEVICE
    memory_format = torch.contiguous_format
    tensor_dtype = TENSOR_DTYPE
    target_dtype = TARGET_DTYPE

    # set random seed
    set_random_seed(45)

    # LOADING IN THE SPLITS
    train_df = pd.read_csv("./data/train_no_outliers.csv")
    val_df = pd.read_csv("./data/val_no_outliers.csv")
    test_df = pd.read_csv("./data/test_no_outliers.csv")

    # 3D DATASET

    # transform_train = get_augmentation((250, 500, 500), True)
    # transform_val = get_augmentation((250, 500, 500))

    traindataset = CancerSphere3DDataset(df=train_df, n_classes=3, reshape_size=(200, 500, 500))
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

    valdataset = CancerSphere3DDataset(df=val_df, n_classes=3, reshape_size=(200, 500, 500))
    valdataloader = nd_Dataloader(
        valdataset,
        batch_size=2,
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

    testdataset = CancerSphere3DDataset(df=test_df, n_classes=3, reshape_size=(200, 500, 500))
    testdataloader = nd_Dataloader(
        testdataset,
        batch_size=1,
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
    # 3D training

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/3D_{now_str}")
    writer_dict = {"writer": writer, "now_str": now_str, "run_dir": f"runs/3D_{now_str}"}

    train3D(traindataloader, valdataloader, "no_outliers_test", 100, 100, 0.0001, writer=writer_dict)

    # evaluate(testdataloader,'./models/model_no_outliers_test.pt', writer=writer_dict)
