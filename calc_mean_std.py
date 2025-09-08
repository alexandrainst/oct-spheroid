import pandas as pd
import torch

from src.data.make_dataset import CustomDataset2D

if __name__ == "__main__":
    train_df = pd.read_csv("./data/train_no_outliers.csv")

    traindataset = CustomDataset2D(df=train_df, n_classes=3, norm=False)
    mean, std = [], []
    for idx in range(len(traindataset)):
        img, label = traindataset.__getitem__(idx)
        mean.append(img.mean(dim=(1, 2)))
        std.append(img.std(dim=(1, 2)))
    mean_final = torch.stack(mean).mean(dim=0)
    std_final = torch.stack(std).mean(dim=0)
    print(mean_final, std_final)
