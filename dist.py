import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.utils.data.distributed

from torchvision import models, datasets, transforms

import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt

import time
"""
torchrun --nproc_per_node=1 dist.py 
    会自动分配环境变量，LOCAL_RANK，每个进程会分配一个LOCAL_RANK
 

"""
def train():
    parser = argparse.ArgumentParser("Mnist Training")
    parser.add_argument("--dataset_path", default="mnist", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    args = parser.parse_args()


    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}"
    dist.init_process_group(backend="nccl", )

    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    model = nn.Sequential(
        models.resnet18(num_classes=64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    model[0].conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)


    dataset = datasets.MNIST(root=args.dataset_path,
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset,  batch_size=args.batch_size, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=device)

    all_losses = []
    print(f"Training on {device}, Rank:{dist.get_rank()}")

    for e in range(0, args.epoch):
        losses = 0.0
        with tqdm(dataloader, unit="train_step") as pbar:
            for idx, (data, label) in enumerate(pbar):
                data, label = data.to(device), label.to(device)
                with torch.amp.autocast(device_type=device.split(":")[0], dtype=torch.float16, ):
                    pred_label = model(data)
                    loss = criterion(pred_label, label)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                losses += loss.item()
                pbar.set_description(f"Device:{device}, Rank:{dist.get_rank()} epoch:[{e + 1}|{args.epoch}] step:[{idx + 1}|{len(dataloader)}] ")
                pbar.set_postfix(loss=f"{loss:.4f}")
        pbar.update()
        all_losses.append(losses)

    if dist.get_rank() == 0 :
        torch.save(model.state_dict(), "model.pth")
        print(f"Model saved by Process ID:{os.getpid()}")

    dist.destroy_process_group()

    x = [i for i in range(args.epoch)]
    plt.plot(x, all_losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mnist Epoch | Loss")
    plt.savefig("loss.png")


if __name__ == "__main__":
    start_time = time.time()
    train()
    print(f"Training Spent {time.time() - start_time:.4f}")