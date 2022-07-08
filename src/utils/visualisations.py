import io
import os

from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import chess
import chess.svg
import chess.pgn
from cairosvg import svg2png


def plot_loss(loss_dir, save_path=None):
    """
    Plots loss changes during training

    Args:
        loss_dir: Path to directory containing CSV files with losses for training
        save_path: If provided, saves plot under given path

    """

    files = sorted(os.listdir(loss_dir))
    losses = np.array([])
    for f in files:
        epoch_losses = pd.read_csv(os.path.join(loss_dir, f))["train_loss"].to_numpy()
        losses = np.concatenate((losses, epoch_losses))
    plt.figure()
    plt.plot(np.linspace(0, len(files), losses.shape[0]), losses)
    plt.title("Training loss")
    plt.xticks(range(0, len(files)+1, 5))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_accuracy(csv, save_path=None):
    """
    Plots accuracy changes during training on test dataset

    Args:
        csv: Path to CSV file with results on test dataset during training
        save_path: If provided, saves plot under given path

    """

    dataframe = pd.read_csv(csv)
    accuracy = dataframe["accuracy"].to_numpy()
    pieces_accuracy = dataframe["pieces_accuracy"].to_numpy()
    plt.figure()
    plt.plot(range(1, accuracy.shape[0]+1), accuracy, label="Accuracy (overall)")
    plt.plot(range(1, accuracy.shape[0]+1), pieces_accuracy, label="Accuracy (pieces)")
    plt.title("Accuracy")
    plt.xticks([1] + list(range(5, accuracy.shape[0]+1, 5)))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_position(fen: str, save_path=None):
    """
    Save given position in FEN format as PNG image under save_path

    Args:
        fen: String with chess position encoded in FEN format
        save_path: If provided, saves plot under given path

    """

    board = chess.Board(fen)
    svg = str(chess.svg.board(board))
    image_bytes = svg2png(svg)
    image = Image.open(io.BytesIO(image_bytes))
    if save_path is None:
        image.show()
    else:
        image.save(save_path)
