import os

from matplotlib import pyplot as plt
import pandas as pd
import torch


def plot_loss(loss_dir):
    files = sorted(os.listdir(loss_dir))
    losses = torch.tensor([])
    for f in files:
        loss_tensor = torch.load(os.path.join(loss_dir, f))
        losses = torch.hstack([losses, loss_tensor])
        print(losses.shape)
    plt.plot(torch.linspace(0, len(files), losses.shape[0]), losses)
    plt.title("Training loss")
    plt.xticks(range(1, len(files)+1))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_accuracy(csv):
    dataframe = pd.read_csv(csv)
    accuracy = dataframe["accuracy"].to_numpy()
    plt.bar(range(1, accuracy.shape[0]+1), accuracy)
    plt.title("Accuracy on test set")
    plt.xticks(range(1, accuracy.shape[0]+1))
    plt.ylim((0, 100))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
