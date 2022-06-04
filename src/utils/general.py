import os
import io

from matplotlib import pyplot as plt
import pandas as pd
import torch
import chess
import chess.svg
import cairosvg

from src.utils.data_transform import board_matrix_to_fen


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
