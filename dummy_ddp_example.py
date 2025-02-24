import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils
import torch.utils.data
from nd_dataloader.examples.example_RK import DummyDatasetClassification3D
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.models.generate_model import generate

EPOCHS = 3
CHANNEL = 1
DIM = 3
BATCH_SIZE = 1
NUM_WORKERS = 10
STEPS_PER_EPOCH = 100
# DEVICE = torch.device("cuda:0")
assert torch.cuda.is_available(), "CUDA not available"
TENSOR_DTYPE = torch.float16
TARGET_DTYPE = torch.long
HEIGHT = 800
WIDTH = 800
DEPTH = 100
DATASET_LENGTH = 400
MEMORY_FORMAT = torch.contiguous_format
NUM_WORKERS = min(12 // torch.cuda.device_count(), 4)
LEARNING_RATE = 0.001


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    setup(rank, world_size)

    model = generate("resnet", 10, 3, 1, 128)  # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    dataset = DummyDatasetClassification3D(
        channel=CHANNEL, height=HEIGHT, width=WIDTH, depth=DEPTH, length=DATASET_LENGTH, tensor_type=TENSOR_DTYPE, target_type=TARGET_DTYPE
    )
    # dataloader = nd_Dataloader(
    #     dataset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     memory_format=MEMORY_FORMAT,
    #     use_prefetcher=True,
    #     mean=dataset.mean,
    #     std=dataset.std,
    #     dim=DIM,
    #     is_training=True,
    #     pin_memmory=False,
    #     tensor_dtype=TENSOR_DTYPE,
    #     target_dtype=TARGET_DTYPE,
    #     batch_norm=True,
    #     distributed=True,
    #     rank=rank,
    #     world_size=world_size
    # )
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    number_of_samples = 0

    # dataloader_iter = iter(dataloader)
    for _ in tqdm(range(EPOCHS), total=EPOCHS):
        ddp_model.train()
        # for _ in tqdm(range(len(dataloader)),total=STEPS_PER_EPOCH, desc="Training"):
        for input, target in tqdm(dataloader, total=STEPS_PER_EPOCH, desc="Training"):
            # input, target = next(dataloader_iter)
            input = input.to(rank)
            target = target.to(rank)
            number_of_samples += input.shape[0]
            # assert input.device == rank, f"input device is {input.device}, expected {rank}"
            # assert target.device == rank, f"target device is {target.device}, expected {rank}"
            # assert input.dtype == TENSOR_DTYPE, f"input dtype is {input.dtype}, expected {TENSOR_DTYPE}"
            # assert target.dtype == TARGET_DTYPE, f"target dtype is {target.dtype}, expected {TARGET_DTYPE}"
            # # forward pass
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = ddp_model(input)
                loss = criterion(output, target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
