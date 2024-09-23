import pandas as pd
import torch
from nd_dataloader import Dataloader as nd_Dataloader

from src.data.make_dataset import (
    CancerSphere3DDataset,  # , CustomDataset, CustomDataset2D
)

# from src.models.generate_model import main
from src.models.train_model import train3D  # , train_multiview

# from torch.utils.data import DataLoader
# from volumentations import *


# from src.models.predict_model import evaluate, evaluate2D


# from torchsummary import summary
# from torch_trandsforms import Compose, RandomFlip, RandomRotate, Resize


# def get_augmentation(patch_size, train=False):
#     if train:
#         return Compose(
#             [
#                 RandomRotate((-15, 15), (0, 0), (0, 0), p=0.5),
#                 # RandomCropFromBorders(crop_value=0.1, p=0.5),
#                 # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
#                 Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
#                 RandomFlip(0, p=0.5),
#                 RandomFlip(1, p=0.5),
#                 RandomFlip(2, p=0.5),
#                 # RandomRotate90((1, 2), p=0.5),
#                 # GaussianNoise(var_limit=(0, 5), p=0.2),
#                 # RandomGamma(gamma_limit=(80, 120), p=0.2),
#             ],
#             p=1.0,
#         )
#     else:
#         return Compose(
#             [
#                 # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
#                 # RandomCropFromBorders(crop_value=0.1, p=0.5),
#                 # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
#                 Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
#                 # Flip(0, p=0.5),
#                 # Flip(1, p=0.5),
#                 # Flip(2, p=0.5),
#                 # RandomRotate90((1, 2), p=0.5),
#                 # GaussianNoise(var_limit=(0, 5), p=0.2),
#                 # RandomGamma(gamma_limit=(80, 120), p=0.2),
#             ],
#             p=1.0,
#         )
# Ok the current loading wall time is

# ~125h for 50 epochs on a 500*500*250 resolution for 1 worker 1 GPU

# ~125h for 50 epochs on a 500*500*250 resolution for 1 worker 1 GPU


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
    # height = 64
    # width = 64
    # depth = 64
    # length = 2**12
    memory_format = torch.contiguous_format
    tensor_dtype = TENSOR_DTYPE
    target_dtype = TARGET_DTYPE

    # LOADING IN THE SPLITS

    # train_df = pd.read_csv("./data/train_split_filtered.csv")
    # val_df = pd.read_csv("./data/val_split_filtered.csv")
    # test_df = pd.read_csv("./data/test_split_filtered.csv")

    train_df = pd.read_csv("./data/filtered_new_train_split.csv")
    val_df = pd.read_csv("./data/filtered_new_val_split.csv")
    test_df = pd.read_csv("./data/filtered_new_test_split.csv")
    ################################################################################################################################################################################################################################

    # 2D DATASET

    # l_dict = {'CTRL':0, "100": 1}

    # traindataset = CustomDataset2D(df = train_df,n_classes=2,labels_dict=l_dict)
    # valdataset = CustomDataset2D(df = val_df,n_classes=2,labels_dict=l_dict)
    # testdataset = CustomDataset2D(df = test_df,n_classes=2,labels_dict=l_dict,return_filename=True)

    # traindataset = CustomDataset2Dslices(train_df,4,2)
    # traindataloader = DataLoader(traindataset,batch_size=5,shuffle=True)
    # # valdataset = CustomDataset2Dslices(val_df,4,2)
    # valdataloader = DataLoader(valdataset,batch_size=2)

    # testdataset = CustomDataset2Dslices(test_df,8,2)
    # testdataloader = DataLoader(testdataset,batch_size=1)

    ################################################################################################################################################################################################################################
    # 3D DATASET

    # transform_train = get_augmentation((250, 500, 500), True)
    # transform_val = get_augmentation((250, 500, 500))

    # traindataset = CustomDataset(df=train_df, n_classes=3,reshape_size=(200,500,500))
    traindataset = CancerSphere3DDataset(df=train_df, n_classes=3, reshape_size=(200, 500, 500))
    # traindataloader = DataLoader(traindataset, batch_size=2,num_workers=0, shuffle=True)
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
    # valdataloader = DataLoader(valdataset, batch_size=2,num_workers=0, shuffle=False)
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

    # testdataset = CustomDataset(df=test_df, n_classes=3,reshape_size=(200,400,400))
    # testdataloader = nd_Dataloader(
    #     testdataset,
    #     batch_size=1,
    #     num_workers=num_workers,
    #     memory_format=memory_format,
    #     use_prefetcher=True,
    #     mean=testdataset.mean,
    #     std=testdataset.std,
    #     dim=DIM,
    #     is_training=False,
    #     pin_memmory=False,
    #     tensor_dtype=torch.float16,
    #     target_dtype=torch.long,
    #     batch_norm=True,
    # )
    # testdataset = CustomDataset(test_df)
    # testdataloader = DataLoader(testdataset, batch_size=1, shuffle=True)
    # dataloader = nd_Dataloader(
    #     testdataset,
    #     batch_size=batch_size,
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

    train3D(traindataloader, valdataloader, "200_500_500_test", 100, 100, 0.0001)

    # evaluate(testdataloader,'./models/model_model_allclasses_v3_nddataloader.pt')
    # evaluate2D(testdataset,'./results/models/model_resnet2D_4slices_fulltraining.pt')

    ################################################################################################################################################################################################################################
    # 2D training
    # print(f"Training dataset size: {len(traindataset)}, validation dataset size {len(valdataset)}")
    # train_multiview(traindataloader, valdataloader, "multiview_4slices_no_aug_25_100_old_data", 500, 500, 0.0001,2)

    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_fulldata_v1.pt',3)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_CTRL_25_v1.pt',2)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_CTRL_100_v1.pt',2)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_25_100_v1.pt',2)

    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_all_classes_fulldata.pt',3)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_CTRL_25_fulldata.pt',2)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_CTRL_100_fulldata.pt',2)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_25_100_fulldata.pt',2)

    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_CTRL_25_old_data.pt',2)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_CTRL_100_old_data.pt',2)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_25_100_old_data.pt',2)
    # evaluate2D(testdataset,'./models/model_multiview_4slices_no_aug_all_classes_old_data.pt',3)

    # img, tar = next(iter(traindataloader))
    # print(img.shape)
    # print(tar)
    # # net = SeparateChannelsResNet18()

    # i = io.imread('./data/20230502/100/2F_8slices.tif')
    # print(i.shape)
    ################################################################################################################################################################################################################################

    # CALCULATING THE NORMALISATION PARAMETERS
    # mean = 0.0
    # std = 0.0
    # nb_samples = 0.0

    # testDataLoader_iter = iter(traindataloader)
    #     # for step, (img, label) in enumerate(valdataload):
    # # images = []
    # for step in tqdm(range(len(traindataloader)), desc="Calculating normalisation parameters", leave=True, position=0, total=len(traindataloader)):
    #     img, label = next(testDataLoader_iter)
    #     # images.append(img)
    #     batch_samples = img.size(0)  # Batch size (number of images in the batch)
    #     # images = img.view(batch_samples, img.size(1), -1)  # Reshape to (batch_size, channels, height*width)
    #     mean += img.mean().item()  # Sum the mean of each channel
    #     std += img.std().item()  # Sum the std of each channel
    #     nb_samples += batch_samples
    # # data_tensor = torch.stack(images)
    # mean /= nb_samples
    # std /= nb_samples

    # print(f"Mean: {mean}")
    # print(f"Std: {std}")
