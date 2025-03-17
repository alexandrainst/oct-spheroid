import pandas as pd
import torch

# from nd_dataloader import Dataloader as nd_Dataloader
# from torch.utils.data import DataLoader

# from src.data.make_dataset import (  # , CustomDataset,
#     CancerSphere3DDataset,
#     CustomDataset2D,
# )
# from src.models.predict_model import evaluate, evaluate2D

# from src.models.generate_model import main
# from src.models.train_model import train3D, train_multiview  # ,

# from volumentations import *


# from torchsummary import summary
# from torch_trandsforms import Compose, RandomFlip, RandomRotate, Resize


if __name__ == "__main__":
    EPOCHS = 50
    CHANNEL = 1
    DIM = 3
    BATCH_SIZE = 2
    NUM_WORKERS = 1
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

    ################################################################################################################################################################################################################################

    # LOADING IN THE SPLITS
    train_df = pd.read_csv("./data/train_no_outliers.csv")
    val_df = pd.read_csv("./data/val_no_outliers.csv")
    test_df = pd.read_csv("./data/test_no_outliers.csv")

    ################################################################################################################################################################################################################################

    # 2D DATASET

    # Depending on the binary classifier you need to adjust the dictionary
    # l_dict = {'CTRL':0, "100": 1}

    # traindataset = CustomDataset2D(df = train_df,n_classes=3)
    # valdataset = CustomDataset2D(df = val_df,n_classes=3)
    # testdataset = CustomDataset2D(df = test_df,n_classes=3,return_filename=True)

    # traindataloader = DataLoader(traindataset,batch_size=5,shuffle=True)
    # valdataloader = DataLoader(valdataset,batch_size=2)
    # testdataloader = DataLoader(testdataset,batch_size=1)

    ################################################################################################################################################################################################################################
    # 3D DATASET

    # transform_train = get_augmentation((250, 500, 500), True)
    # transform_val = get_augmentation((250, 500, 500))

    # traindataset = CancerSphere3DDataset(df=train_df, n_classes=3, reshape_size=(200, 500, 500))
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

    # valdataset = CancerSphere3DDataset(df=val_df, n_classes=3, reshape_size=(200, 500, 500))
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

    # testdataset = CancerSphere3DDataset(df=test_df, n_classes=3, reshape_size=(200, 500, 500))
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

    ################################################################################################################################################################################################################################
    # TESTING

    # number_of_samples = 0
    # total_start = time.time()
    # traindataloader_iter = iter(traindataloader)
    # for _ in tqdm(range(1), total=1):
    #     start = time.time()
    #     for _ in tqdm(range(len(traindataloader)),total=len(traindataloader)):
    #         input, target = next(traindataloader_iter)
    #         input = input.to(TENSOR_DTYPE)
    #         number_of_samples += input.shape[0]
    #         assert input.device == device, f"input device is {input.device}, expected {device}"
    #         assert target.device == device, f"target device is {target.device}, expected {device}"
    #         assert input.dtype == tensor_dtype, f"input dtype is {input.dtype}, expected {tensor_dtype}"
    #         assert target.dtype == target_dtype, f"target dtype is {target.dtype}, expected {target_dtype}"
    #         # # forward pass
    #         # with torch.cuda.amp.autocast(dtype=torch.float16):
    #         #     output = model(input)
    #         #     loss = criterion(output, target)
    #         # optimizer.zero_grad()
    #         # scaler.scale(loss).backward()
    #         # scaler.step(optimizer)
    #         # scaler.update()

    #     # Do something here with input and target
    #     end = time.time()
    #     print("nd_dataloader time:", end - start, "samples:", number_of_samples)
    # total_end = time.time()
    # # log the total time
    # # log(total_end - total_start, "nd", batch_size, height, width, depth, length, now_str=now_str, memory_format=memory_format, epochs=EPOCHS, tensor_dtype=tensor_dtype, target_dtype=target_dtype)
    # print("total time:", total_end - total_start, "samples:", number_of_samples)

    ################################################################################################################################################################################################################################

    # 3D training

    # train3D(traindataloader, valdataloader, "no_outliers_test", 100, 100, 0.0001)

    # evaluate(testdataloader,'./models/model_no_outliers_test.pt')

    ################################################################################################################################################################################################################################
    # 2D training
    # print(f"Training dataset size: {len(traindataset)}, validation dataset size {len(valdataset)}")
    # train_multiview(traindataloader, valdataloader, "multiview_4slices_no_outliers", 500, 500, 0.0001,3)

    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_outliers.pt',3)

    ################################################################################################################################################################################################################################
