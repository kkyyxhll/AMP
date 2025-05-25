import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
from torchvision import models, datasets, transforms

import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mnist Training")
    parser.add_argument("--dataset_path", default="mnist", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--epoch",default=10, type=int)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    model = nn.Sequential(
        models.resnet18(num_classes=64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    model[0].conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    model = model.to(device)

    dataset = datasets.MNIST(root=args.dataset_path,
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)

    dataloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=device)

    all_losses = []

    for e in range(0, args.epoch):
        losses = 0.0
        with tqdm(dataloader, unit="train_step") as pbar:
            for idx, (data, label) in enumerate(pbar):
                data, label = data.to(device), label.to(device)
                with torch.amp.autocast(device_type=device, dtype=torch.float16, ):
                    pred_label = model(data)
                    loss = criterion(pred_label, label)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                losses += loss.item()
                pbar.set_description(f"epoch:[{e+1}|{args.epoch}] step:[{idx+1}|{len(dataloader)}] ")
                pbar.set_postfix(loss=f"{loss:.4f}")
        pbar.update()
        all_losses.append(losses)

    x = [i for i in range(args.epoch)]
    plt.plot(x, all_losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Mnist Epoch | Loss")
    plt.show()