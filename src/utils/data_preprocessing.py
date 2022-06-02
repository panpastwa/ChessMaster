import os

import torch
from matplotlib import pyplot as plt

from src.utils.data_transform import fen_to_board_matrix


def prepare_fen_tensor(fen_filename):

    batch_size = 2**20
    position_tensors = torch.empty((batch_size, 8, 8), dtype=torch.uint8)
    with open(fen_filename) as mf:
        for i, line in enumerate(mf):
            fen = line.strip()
            if fen:
                position_tensor = fen_to_board_matrix(fen)
                position_tensors[i % batch_size] = position_tensor
                if (i+1) % batch_size == 0:
                    part = (i+1) // batch_size
                    print(f"Exported part: {part:<3}")
                    path = f"/media/panpastwa/Vincent/Datasets/Chess/BoardMatrices/part{part:03}.pt"
                    torch.save(position_tensors, path)


def transform_tensors():
    path = "/media/panpastwa/Vincent/Datasets/Chess/BoardMatrices"
    out_path = "/media/panpastwa/Vincent/Datasets/Chess/BoardMatrices2"
    tensors = []
    for i, f in enumerate(sorted(os.listdir(path))):
        tensors.append(torch.load(os.path.join(path, f)))
        if (i+1) % 5 == 0:
            print((i+1)//5)
            torch.save(torch.vstack(tensors), os.path.join(out_path, f"part{(i+1)//5:03}.pt"))
            tensors.clear()


def x():

    path = "/media/panpastwa/Vincent/Datasets/Chess/BoardMatrices2"
    path2 = "/media/panpastwa/Vincent/Datasets/Chess/BoardMatricesSplitted"

    for f in os.listdir(path):
        os.link(os.path.join(path, f), os.path.join(path2, f))


losses = torch.load("/home/panpastwa/Projects/ChessMasterThesis/runs/001/train_losses/epoch_02.pt")
plt.plot(range(losses.numel()), losses)
plt.show()
