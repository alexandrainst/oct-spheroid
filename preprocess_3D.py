import argparse

import pandas as pd

from src.data import CancerSphere3DDataset
from src.utils import config_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Dataset Preprocessing")
    parser.add_argument("--config", type=str, default="config/experiments/config_3D.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = config_handler(args.config)

    HEIGHT = config["DATASET"]["HEIGHT"]
    WIDTH = config["DATASET"]["WIDTH"]
    DEPTH = config["DATASET"]["DEPTH"]

    train_df = pd.read_csv("./data/train_no_outliers.csv")
    val_df = pd.read_csv("./data/val_no_outliers.csv")

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
    else:
        raise ValueError(f"Unknown task {config['GENERAL']['TASK']}")

    traindataset = CancerSphere3DDataset(
        df=train_df,
        n_classes=config["MODEL"]["NUMBER_OF_CLASSES"],
        reshape_size=(DEPTH, HEIGHT, WIDTH),
        task=config["GENERAL"]["TASK"],
        store_pp=config["DATASET"]["STORE_PP"],
        store_pp_path=config["DATASET"]["STORE_PP_PATH"],
        labels_dict=labels_dict,
    )
    traindataset.preprocess()

    valdataset = CancerSphere3DDataset(
        df=val_df,
        n_classes=config["MODEL"]["NUMBER_OF_CLASSES"],
        reshape_size=(DEPTH, HEIGHT, WIDTH),
        task=config["GENERAL"]["TASK"],
        store_pp=config["DATASET"]["STORE_PP"],
        store_pp_path=config["DATASET"]["STORE_PP_PATH"],
        labels_dict=labels_dict,
    )
    valdataset.preprocess()
