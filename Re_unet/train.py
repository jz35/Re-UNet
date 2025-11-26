import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader

from preprocess import DriveDataset
from loss import DiceLoss, DiceBCELoss
from utils import (
    seeding,
    create_dir,
    epoch_time,
    binary_accuracy,
    plot_loss_accuracy,
)
from Re_unet import ReMoEUNet


def _unwrap_outputs(outputs, device):
    if isinstance(outputs, tuple):
        return outputs
    return outputs, torch.tensor(0.0, device=device)


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(x)
        y_pred, aux_loss = _unwrap_outputs(outputs, device)
        loss = loss_fn(y_pred, y) + aux_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += binary_accuracy(y_pred, y)

    epoch_loss = epoch_loss / len(loader)
    epoch_acc = epoch_acc / len(loader)
    return epoch_loss, epoch_acc


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            outputs = model(x)
            y_pred, aux_loss = _unwrap_outputs(outputs, device)
            loss = loss_fn(y_pred, y) + aux_loss
            epoch_loss += loss.item()
            epoch_acc += binary_accuracy(y_pred, y)

        epoch_loss = epoch_loss / len(loader)
        epoch_acc = epoch_acc / len(loader)
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("data/train/image/*"))
    train_y = sorted(glob("data/train/mask/*"))

    valid_x = sorted(glob("data/test/image/*"))
    valid_y = sorted(glob("data/test/mask/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReMoEUNet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f} | Acc: {train_acc:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f} | Acc: {valid_acc:.3f}\n'
        print(data_str)

    plot_path = os.path.join("files", "re_unet_train_curves.png")
    plot_loss_accuracy(
        train_losses, valid_losses, train_accs, valid_accs, plot_path, "Re-UNet"
    )