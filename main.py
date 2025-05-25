import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets

import argparse

import math

import matplotlib.pyplot as plt
from tqdm import tqdm
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def train(color, plt_label, grad_clip=False, amp=False):
    print(f"grad_clip: {grad_clip}, amp: {amp}")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(device = "cuda", enabled=amp)

    model.train()

    all_losses = []

    for e in range(0, args.epoch):
        losses = 0.0
        with tqdm(train_dataloader, unit="train") as pbar:
            for idx, (data, label) in enumerate(pbar):
                optimizer.zero_grad()
                data, label = data.to(device), label.to(device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    pred_label = model(data)
                    loss = criterion(pred_label, label)
                    scaler.scale(loss).backward()

                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()


                losses += loss.item()
                pbar.set_description(f"epoch:[{e + 1}|{args.epoch}] step:[{idx + 1}|{len(train_dataloader)}]")
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        losses = losses / len(train_dataloader)
        all_losses.append(losses)

    x = [i for i in range(args.epoch)]

    plt.plot(x, all_losses, color=color, label=plt_label)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mnist Test")
    parser.add_argument("--root_path", type=str, default="mnist")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    print(f"device:{device}")
    if device == "cuda":
        torch.cuda.manual_seed_all(42)

    train_dataset = datasets.MNIST(root=args.root_path, train=True, transform=transforms.ToTensor(), download=True)

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)


    train(color="red", plt_label="nothing", )


    plt.legend()
    plt.title("Loss | Epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
