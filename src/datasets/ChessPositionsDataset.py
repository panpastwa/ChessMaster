import torch

from torch.utils.data import Dataset


class ChessPositionsDataset(Dataset):

    def __init__(self, path):
        self.positions = torch.load(path)

    def __getitem__(self, item):
        return self.positions[item]

    def __len__(self):
        return self.positions.shape[0]
