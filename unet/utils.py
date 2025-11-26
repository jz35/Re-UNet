import os
import time
import random
import numpy as np
import cv2
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(logits, targets):
    """Return mean pixel accuracy for binary segmentation logits."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == targets).float().mean().item()


def plot_loss_accuracy(train_losses, valid_losses, train_accs, valid_accs, save_path, title):
    """Plot train/valid loss & accuracy curves."""
    if not train_losses or not valid_losses:
        return

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, valid_accs, label="Valid Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_test_curves(losses, accs, save_path, title):
    """Plot per-sample loss & accuracy curves for test set."""
    if not losses or not accs:
        return

    indices = range(1, len(losses) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(indices, losses, label="Sample Loss")
    plt.xlabel("Sample")
    plt.ylabel("Loss")
    plt.title(f"{title} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(indices, accs, label="Sample Accuracy")
    plt.xlabel("Sample")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()