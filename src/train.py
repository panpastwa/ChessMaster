from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ChessPositionsDataset import ChessPositionsDataset


def train(model, files, batch_size, device, optimizer, loss_function, positions_per_file, regularization=False):
    pbar = tqdm(total=int(len(files) * positions_per_file / batch_size))
    losses = []
    for file in files:
        dataset = ChessPositionsDataset(file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        for batch in dataloader:
            batch = batch.long().to(device)
            one_hot_batch = one_hot(batch, num_classes=13).flatten(start_dim=1).float()
            optimizer.zero_grad()
            y = model(one_hot_batch)
            pred = y.reshape(-1, 8, 8, 13).permute(0, 3, 1, 2)
            loss = loss_function(pred, batch)
            if regularization:
                # TODO: add regularization
                pass
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pbar.update()
            pbar.set_description(f"Loss: {loss:4.3f}")
    pbar.close()
    return losses
