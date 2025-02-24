import torch
from nd_dataloader import Dataloader as nd_Dataloader
from nd_dataloader.examples.example_RK import DummyDatasetClassification3D
from tqdm import tqdm

from src.models.generate_model import generate

EPOCHS = 10
CHANNEL = 1
DIM = 3
BATCH_SIZE = 1
NUM_WORKERS = 10
STEPS_PER_EPOCH = 100
DEVICE = torch.device("cuda:0")
assert torch.cuda.is_available(), "CUDA not available"
TENSOR_DTYPE = torch.float16
TARGET_DTYPE = torch.long
HEIGHT = 900
WIDTH = 900
DEPTH = 100
DATASET_LENGTH = 400
MEMORY_FORMAT = torch.contiguous_format
NUM_WORKERS = 2
LEARNING_RATE = 0.001


def main():
    model = generate("resnet", 10, 3, 1, 128)  # net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    dataset = DummyDatasetClassification3D(
        channel=CHANNEL, height=HEIGHT, width=WIDTH, depth=DEPTH, length=DATASET_LENGTH, tensor_type=TENSOR_DTYPE, target_type=TARGET_DTYPE
    )
    dataloader = nd_Dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        memory_format=MEMORY_FORMAT,
        use_prefetcher=True,
        mean=dataset.mean,
        std=dataset.std,
        dim=DIM,
        is_training=True,
        pin_memmory=False,
        tensor_dtype=TENSOR_DTYPE,
        target_dtype=TARGET_DTYPE,
        batch_norm=True,
    )
    number_of_samples = 0

    dataloader_iter = iter(dataloader)
    for _ in tqdm(range(EPOCHS), total=EPOCHS):
        model.train()
        for _ in tqdm(range(len(dataloader)), total=STEPS_PER_EPOCH, desc="Training"):
            input, target = next(dataloader_iter)
            number_of_samples += input.shape[0]
            assert input.device == DEVICE, f"input device is {input.device}, expected {DEVICE}"
            assert target.device == DEVICE, f"target device is {target.device}, expected {DEVICE}"
            assert input.dtype == TENSOR_DTYPE, f"input dtype is {input.dtype}, expected {TENSOR_DTYPE}"
            assert target.dtype == TARGET_DTYPE, f"target dtype is {target.dtype}, expected {TARGET_DTYPE}"
            # # forward pass
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(input)
                loss = criterion(output, target)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


if __name__ == "__main__":
    main()
