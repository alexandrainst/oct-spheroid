import argparse

import pandas as pd

from src.data import MultiviewDataset
from src.utils import config_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Dataset Preprocessing")
    parser.add_argument("--config", type=str, default="config/experiments/config_2D.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = config_handler(args.config)

    train_df = pd.read_csv("./data/train_no_outliers.csv")
    val_df = pd.read_csv("./data/val_no_outliers.csv")

    traindataset = MultiviewDataset(
        df=train_df, n_classes=config["MODEL"]["NUMBER_OF_CLASSES"],
        transform=None,
        store_pp_path=config["DATASET"]["STORE_PP_PATH"],
        )
    traindataset.preprocess()

    valdataset = MultiviewDataset(
        df=val_df,
        n_classes=config["MODEL"]["NUMBER_OF_CLASSES"],
        transform=None,
        store_pp_path=config["DATASET"]["STORE_PP_PATH"],
    )
    valdataset.preprocess()
