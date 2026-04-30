import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# from volumentations import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2

from src.data.dataset_multiview import MultiviewDataset

# from src.models.generate_model import main
from src.models.train_model import train_multiview
from src.utils import MultiLibDumper, config_handler, set_random_seed




if __name__ == "__main__":
    # rad yaml file
    config = config_handler("config/experiments/config_2D.yaml")

    print(config)
    DEVICE = torch.device("cuda:1")
    assert torch.cuda.is_available(), "CUDA not available"
    
    # set random seed
    set_random_seed(45)

    # LOADING IN THE SPLITS
    train_df = pd.read_csv("./data/train_no_outliers.csv")
    val_df = pd.read_csv("./data/val_no_outliers.csv")

    transforms_train = v2.Compose(
        [
            v2.Normalize(mean=[12.4361, 12.4864, 12.4278, 12.3149], std=[10.7506, 10.7760, 9.8911, 9.9736]),
        ]
    )
    transforms_val = v2.Compose(
        [
            v2.Normalize(mean=[12.4361, 12.4864, 12.4278, 12.3149], std=[10.7506, 10.7760, 9.8911, 9.9736]),
        ]
    )

    traindataset = MultiviewDataset(
        df=train_df, n_classes=config["MODEL"]["NUMBER_OF_CLASSES"],
        transform=transforms_train,
        store_pp=config["DATASET"]["STORE_PP"],
        store_pp_path=config["DATASET"]["STORE_PP_PATH"],
        )
    traindataloader = DataLoader(
        traindataset, batch_size=config["DATALOADER"]["TRAIN"]["BATCH_SIZE"], shuffle=True, num_workers=config["DATALOADER"]["TRAIN"]["NUM_WORKERS"]
    )

    valdataset = MultiviewDataset(
        df=val_df,
        n_classes=config["MODEL"]["NUMBER_OF_CLASSES"],
        transform=transforms_val,
        store_pp=config["DATASET"]["STORE_PP"],
        store_pp_path=config["DATASET"]["STORE_PP_PATH"],
    )
    valdataloader = DataLoader(
        valdataset, batch_size=config["DATALOADER"]["VAL"]["BATCH_SIZE"], shuffle=False, num_workers=config["DATALOADER"]["VAL"]["NUM_WORKERS"]
    )

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"runs/2D_{now_str}")
    writer_dict = {"writer": writer, "now_str": now_str, "run_dir": f"runs/2D_{now_str}"}
    # copy config to run_dir
    with open(f"{writer_dict['run_dir']}/config.yaml", "w") as file:
        yaml.dump(config, file, Dumper=MultiLibDumper, default_flow_style=False)
    train_multiview(traindataloader, valdataloader, writer=writer_dict, config=config)

    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_outliers.pt',3, writer=writer_dict)
