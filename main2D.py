import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# from volumentations import *
from torch.utils.tensorboard import SummaryWriter

from src.data.make_dataset import CustomDataset2D

# from src.models.generate_model import main
from src.models.train_model import train_multiview

# from nd_dataloader import Dataloader as nd_Dataloader


# from src.models.predict_model import evaluate, evaluate2D


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
    DIM = 2
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    STEPS_PER_EPOCH = 100
    DEVICE = torch.device("cuda:1")
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

    traindataset = CustomDataset2D(df=train_df, n_classes=3)
    # traindataloader = nd_Dataloader(
    #     traindataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     memory_format=memory_format,
    #     use_prefetcher=True,
    #     mean=traindataset.mean,
    #     std=traindataset.std,
    #     dim=DIM,
    #     is_training=True,
    #     pin_memmory=False,
    #     tensor_dtype=torch.float16,
    #     target_dtype=torch.long,
    #     batch_norm=True,
    # )
    traindataloader = DataLoader(traindataset, batch_size=5, shuffle=True)

    valdataset = CustomDataset2D(df=val_df, n_classes=3)
    # valdataloader = nd_Dataloader(
    #     valdataset,
    #     batch_size=2,
    #     num_workers=num_workers,
    #     memory_format=memory_format,
    #     use_prefetcher=True,
    #     mean=valdataset.mean,
    #     std=valdataset.std,
    #     dim=DIM,
    #     is_training=True,
    #     pin_memmory=False,
    #     tensor_dtype=torch.float16,
    #     target_dtype=torch.long,
    #     batch_norm=True,
    # )
    valdataloader = DataLoader(valdataset, batch_size=2)

    testdataset = CustomDataset2D(df=test_df, n_classes=3, return_filename=True)
    # testdataloader = nd_Dataloader(
    #     testdataset,
    #     batch_size=1,
    #     num_workers=num_workers,
    #     memory_format=memory_format,
    #     use_prefetcher=True,
    #     mean=testdataset.mean,
    #     std=testdataset.std,
    #     dim=DIM,
    #     is_training=True,
    #     pin_memmory=False,
    #     tensor_dtype=torch.float16,
    #     target_dtype=torch.long,
    #     batch_norm=True,
    # )
    testdataloader = DataLoader(testdataset, batch_size=1)
    # 3D training

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/2D_{now_str}")
    writer_dict = {"writer": writer, "now_str": now_str, "run_dir": f"runs/2D_{now_str}"}

    train_multiview(traindataloader, valdataloader, "multiview_4slices_no_outliers", 500, 500, 0.0001, 3, writer=writer_dict)

    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_outliers.pt',3, writer=writer_dict)
