from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from src.datasets.ChessPositionsDataset import ChessPositionsDataset


def train(model, files, batch_size, device, optimizer, loss_function, positions_per_file,
          regularization=False, reg_weight=0.01):

    pbar = tqdm(total=int(len(files) * positions_per_file / batch_size))
    losses, reg_losses = [], []
    reg_weight = torch.tensor(reg_weight, device=device)
    reg_desc_text = ''
    for file in files:
        dataset = ChessPositionsDataset(file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        for batch in dataloader:
            batch = batch.long().to(device)
            one_hot_batch = one_hot(batch, num_classes=13).flatten(start_dim=1).float()
            optimizer.zero_grad()
            y = model(one_hot_batch)
            pred = y.reshape(-1, 8, 8, 13)
            loss = loss_function(pred.permute(0, 3, 1, 2), batch)
            losses.append(loss.item())

            if regularization:
                num_pieces_batch = batch.count_nonzero(dim=(1, 2))
                pred_pieces = pred.argmax(dim=-1)
                num_pieces_pred = pred_pieces.count_nonzero(dim=(1, 2))
                reg_loss = reg_weight * (num_pieces_pred - num_pieces_batch).abs().sum() / batch_size
                reg_losses.append(reg_loss.item())
                reg_desc_text = f" (Loss: {loss:4.3f} | Regularization loss: {reg_loss:4.3f})"
                loss += reg_loss

            desc_text = f"Total loss: {loss:4.3f}" + reg_desc_text
            loss.backward()
            optimizer.step()
            pbar.update()
            pbar.set_description(desc_text)
    pbar.close()
    return losses, reg_losses
