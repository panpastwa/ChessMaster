import os

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import chess
import chess.svg
import cairosvg

from src.utils.data_transform import board_matrix_to_fen


def plot_loss(loss_dir, save_path=None):
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


def visualise_predictions(target, output):
    # TODO: improve
    assert target.shape == output.shape
    for counter, (x, y) in enumerate(zip(target, output)):
        x, y = chess.Board(board_matrix_to_fen(x)), chess.Board(board_matrix_to_fen(y))
        # cairosvg.svg2png(io.BytesIO(chess.svg.board(x)))  # TODO: export png
        o = chess.svg.board(x)
        with open(f"board{counter}_x.svg", 'w') as mf:
            mf.write(str(chess.svg.board(x)))
        with open(f"board{counter}_y.svg", 'w') as mf:
            mf.write(str(chess.svg.board(y)))
