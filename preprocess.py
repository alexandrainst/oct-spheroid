import pandas as pd

from src.data.make_dataset import CancerSphere3DDataset

if __name__ == "__main__":
    HEIGHT = 500
    WIDTH = 500
    DEPTH = 200

    train_df = pd.read_csv("./data/filtered_new_train_split.csv")
    val_df = pd.read_csv("./data/filtered_new_val_split.csv")

    traindataset = CancerSphere3DDataset(df=train_df, n_classes=3, reshape_size=(DEPTH, HEIGHT, WIDTH))
    traindataset.preprocess()

    valdataset = CancerSphere3DDataset(df=val_df, n_classes=3, reshape_size=(DEPTH, HEIGHT, WIDTH))
    valdataset.preprocess()
