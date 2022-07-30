import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.ChessPositionsDataset import ChessPositionsDataset


def validate(model, files, batch_size, device, loss_function, positions_per_file):

    test_loss = torch.tensor([0.0]).to(device)
    total_correct = torch.tensor([0], dtype=torch.long).to(device)
    total_pieces_correct = torch.tensor([0], dtype=torch.long).to(device)
    total_pieces = torch.tensor([0], dtype=torch.long).to(device)
    pbar = tqdm(total=int(len(files) * positions_per_file / batch_size))
    with torch.no_grad():
        for file in files:
            dataset = ChessPositionsDataset(file)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
            for batch in dataloader:
                batch = batch.long().to(device)
                one_hot_batch = one_hot(batch, num_classes=13).flatten(start_dim=1).float()
                y = model(one_hot_batch)
                pred = y.reshape(-1, 8, 8, 13)
                loss = loss_function(pred.permute(0, 3, 1, 2), batch)
                test_loss += loss
                pbar.update()
                pbar.set_description(f"Loss: {loss:4.3f}")

                total_batch = batch.numel()
                pieces_batch = (batch > 0).sum()
                empty_squares_batch = (batch == 0).sum()

                pred = pred.argmax(dim=-1)
                correct = (pred == batch)
                overall_correct = correct.sum()
                overall_incorrect = (~correct).sum()
                pieces_correct = (pred[correct] > 0).sum()
                pieces_incorrect = (pred[~correct] > 0).sum()
                empty_squares_correct = (pred[correct] == 0).sum()
                empty_squares_incorrect = (pred[~correct] == 0).sum()

                # print(f"Squares in batch: {total_batch}")
                # print(f"Pieces total: {pieces_batch}")
                # print(f"Empty squares total: {empty_squares_batch}")
                # print(f"Correct: {overall_correct}")
                # print(f"Incorrect: {overall_incorrect}")
                # print(f"Pieces correct: {pieces_correct}")
                # print(f"Pieces incorrect: {pieces_incorrect}")
                # print(f"Empty squares correct: {empty_squares_correct}")
                # print(f"Empty squares incorrect: {empty_squares_incorrect}")

                total_correct += overall_correct
                total_pieces_correct += pieces_correct
                total_pieces += pieces_batch

    pbar.close()
    mean_test_loss = (test_loss / (len(files) * positions_per_file / batch_size)).item()
    mean_accuracy = (100 * total_correct / (len(files) * positions_per_file * 8 * 8)).item()
    mean_pieces_accuracy = (100 * total_pieces_correct / total_pieces).item()
    return mean_test_loss, mean_accuracy, mean_pieces_accuracy
