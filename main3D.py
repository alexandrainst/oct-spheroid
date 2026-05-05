import argparse
from datetime import datetime

import pandas as pd
import torch
import yaml
from nd_dataloader import Dataloader as nd_Dataloader
from torch.utils.tensorboard import SummaryWriter

from oct.data.dataset_3D import CancerSphere3DDataset
from oct.models.train_model import train3D
from oct.utils import MultiLibDumper, config_handler, set_random_seed

if __name__ == "__main__":
    # read yaml file
    parser = argparse.ArgumentParser(description="3D Model Training")
    parser.add_argument("--config", type=str, default="config/experiments/config_3D.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = config_handler(args.config)

    print(config)
    assert torch.cuda.is_available(), "CUDA not available"

    # set random seed
    set_random_seed(45)

    # LOADING IN THE SPLITS
    train_df = pd.read_csv("./data/train_no_outliers.csv")
    val_df = pd.read_csv("./data/val_no_outliers.csv")
    test_df = pd.read_csv("./data/test_no_outliers.csv")

    # label_dict # TODO move to config
    if config["GENERAL"]["TASK"] == "classification":
        labels_dict = {
            "CTRL": 0,
            "25": 1,
            "100": 2,
        }
    elif config["GENERAL"]["TASK"] == "regression":
        labels_dict = {
            "CTRL": 0.0,
            "25": 0.25,
            "100": 1.0,
        }

    traindataset = CancerSphere3DDataset(
        df=train_df,
        n_classes=config["MODEL"]["NUMBER_OF_CLASSES"],
        reshape_size=(200, 500, 500),
        task=config["GENERAL"]["TASK"],
        store_pp=config["DATASET"]["STORE_PP"],
        store_pp_path=config["DATASET"]["STORE_PP_PATH"],
        labels_dict=labels_dict,
        normalize=not config["DATALOADER"]["USE_ND_DATALOADER"],  # if using nd_dataloader, we will normalize in the dataloader
        mean=config["DATASET"]["MEAN"],
        std=config["DATASET"]["STD"],
    )
    if config["DATALOADER"]["USE_ND_DATALOADER"]:
        traindataloader = nd_Dataloader(
            traindataset,
            batch_size=config["DATALOADER"]["TRAIN"]["BATCH_SIZE"],
            num_workers=config["DATALOADER"]["TRAIN"]["NUM_WORKERS"],
            memory_format=config["DATASET"]["MEMORY_FORMAT"],
            use_prefetcher=True,
            mean=traindataset.mean,
            std=traindataset.std,
            dim=config["DATASET"]["DIM"],
            is_training=True,
            pin_memmory=config["DATALOADER"]["TRAIN"]["PIN_MEMORY"],
            tensor_dtype=config["DATASET"]["TENSOR_DTYPE"],
            target_dtype=config["DATASET"]["TARGET_DTYPE"],
            batch_norm=True,
        )
    else:
        traindataloader = torch.utils.data.DataLoader(
            traindataset,
            batch_size=config["DATALOADER"]["TRAIN"]["BATCH_SIZE"],
            shuffle=True,
            num_workers=config["DATALOADER"]["TRAIN"]["NUM_WORKERS"],
            pin_memory=config["DATALOADER"]["TRAIN"]["PIN_MEMORY"],
        )

    valdataset = CancerSphere3DDataset(
        df=val_df,
        n_classes=config["MODEL"]["NUMBER_OF_CLASSES"],
        reshape_size=(200, 500, 500),
        task=config["GENERAL"]["TASK"],
        labels_dict=labels_dict,
        normalize=not config["DATALOADER"]["USE_ND_DATALOADER"],  # if using nd_dataloader, we will normalize in the dataloader
        mean=config["DATASET"]["MEAN"],
        std=config["DATASET"]["STD"],
    )
    if config["DATALOADER"]["USE_ND_DATALOADER"]:
        valdataloader = nd_Dataloader(
            valdataset,
            batch_size=config["DATALOADER"]["VAL"]["BATCH_SIZE"],
            num_workers=config["DATALOADER"]["VAL"]["NUM_WORKERS"],
            memory_format=config["DATASET"]["MEMORY_FORMAT"],
            use_prefetcher=True,
            mean=valdataset.mean,
            std=valdataset.std,
            dim=config["DATASET"]["DIM"],
            is_training=False,
            pin_memmory=config["DATALOADER"]["VAL"]["PIN_MEMORY"],
            tensor_dtype=config["DATASET"]["TENSOR_DTYPE"],
            target_dtype=config["DATASET"]["TARGET_DTYPE"],
            batch_norm=True,
        )
    else:
        valdataloader = torch.utils.data.DataLoader(
            valdataset,
            batch_size=config["DATALOADER"]["VAL"]["BATCH_SIZE"],
            shuffle=False,
            num_workers=config["DATALOADER"]["VAL"]["NUM_WORKERS"],
            pin_memory=config["DATALOADER"]["VAL"]["PIN_MEMORY"],
        )

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    if config["GENERAL"]["TASK"] == "classification":
        writer = SummaryWriter(f"runs/3D_{now_str}")
        writer_dict = {"writer": writer, "now_str": now_str, "run_dir": f"runs/3D_{now_str}"}
    elif config["GENERAL"]["TASK"] == "regression":
        writer = SummaryWriter(f"runs/3D_regression_{now_str}")
        writer_dict = {"writer": writer, "now_str": now_str, "run_dir": f"runs/3D_regression_{now_str}"}
    else:
        raise ValueError(f"Unknown task {config['GENERAL']['TASK']}")
    # copy config to run_dir
    with open(f"{writer_dict['run_dir']}/config.yaml", "w") as file:
        yaml.dump(config, file, Dumper=MultiLibDumper, default_flow_style=False)

    train3D(traindataloader, valdataloader, writer=writer_dict, config=config)
