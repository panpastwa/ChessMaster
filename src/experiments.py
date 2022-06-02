import os
import logging
import time

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
import pandas as pd
from tqdm import tqdm

from src.models.SimpleAutoEncoder import SimpleAutoEncoder
from src.datasets.ChessPositionsDataset import ChessPositionsDataset
from src.configs.autoencoder_100 import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create new run dir
last_dir = max(os.listdir("runs"))
project_dir = f"runs/{int(last_dir) + 1:03}"
os.mkdir(project_dir)
os.mkdir(os.path.join(project_dir, "train_losses"))
os.mkdir(os.path.join(project_dir, "test_losses"))
os.mkdir(os.path.join(project_dir, "weights"))

# Set up logger
logging.basicConfig(format='', level=logging.DEBUG, handlers=[
    logging.StreamHandler(),
    logging.FileHandler(os.path.join(project_dir, "run.log"))
])
logging.info(f"Saving results to {project_dir}\n")

logging.info(f"DEVICE: {DEVICE}\n"
             f"BATCH SIZE: {BATCH_SIZE}\n"
             f"HIDDEN SIZE: {HIDDEN_SIZE}\n"
             f"EPOCHS: {EPOCHS}\n"
             f"LEARNING RATE: {LEARNING_RATE}\n")

autoencoder = SimpleAutoEncoder(64 * 13, HIDDEN_SIZE).to(DEVICE)
train_dir = "/media/panpastwa/Vincent/Datasets/Chess/BoardMatricesSplitted/train"
test_dir = "/media/panpastwa/Vincent/Datasets/Chess/BoardMatricesSplitted/test"

train_files = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]
test_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir)]
positions_per_file = torch.load(train_files[0]).shape[0]

logging.info(f"{len(train_files)} files will be used for training and {len(test_files)} will be used for testing. "
             f"Each file contain {positions_per_file} postions.\n")

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(autoencoder.parameters(), lr=LEARNING_RATE)

times, accuracies = [], []
best_test_loss = None

for epoch in range(EPOCHS):

    start_time = time.time()
    train_loss = 0.0
    test_loss = 0.0
    test_correct = 0
    train_losses = []
    test_losses = []

    pbar = tqdm(train_files)

    for file in pbar:
        dataset = ChessPositionsDataset(file)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        file_losses = []
        for batch in dataloader:
            batch = batch.long().to(DEVICE)
            one_hot_batch = one_hot(batch, num_classes=13).flatten(start_dim=1).float()
            optimizer.zero_grad()
            y = autoencoder(one_hot_batch)
            pred = y.reshape(-1, 8, 8, 13).permute(0, 3, 1, 2)
            loss = loss_function(pred, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss
            file_losses.append(loss)
            pbar.set_description(f"Loss: {loss:4.3f}")
        train_losses.append(torch.tensor(file_losses).mean())
    pbar.close()

    pbar = tqdm(test_files)
    with torch.no_grad():
        for file in pbar:
            dataset = ChessPositionsDataset(file)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2)
            file_losses = []
            for batch in dataloader:
                batch = batch.long().to(DEVICE)
                one_hot_batch = one_hot(batch, num_classes=13).flatten(start_dim=1).float()
                y = autoencoder(one_hot_batch)
                pred = y.reshape(-1, 8, 8, 13)
                loss = loss_function(pred.permute(0, 3, 1, 2), batch)
                test_loss += loss
                file_losses.append(loss)
                pbar.set_description(f"Loss: {loss:4.3f}")

                pred = pred.argmax(dim=-1)
                correct = (pred == batch).sum()
                test_correct += correct
            test_losses.append(torch.tensor(file_losses).mean())

    mean_training_loss = train_loss / (len(train_files) * positions_per_file / BATCH_SIZE)
    mean_test_loss = test_loss / (len(test_files) * positions_per_file / BATCH_SIZE)
    mean_accuracy = (100 * test_correct / (len(test_files) * positions_per_file * 8 * 8)).item()
    times.append(time.time() - start_time)
    accuracies.append(mean_accuracy)

    logging.info(f"Epoch: {epoch + 1:3d} | "
                 f"Training loss: {mean_training_loss:.3f} | "
                 f"Test loss: {mean_test_loss:.3f} | "
                 f"Accuracy: {mean_accuracy:.3f}")

    torch.save(torch.tensor(train_losses), os.path.join(project_dir, "train_losses", f"epoch_{epoch+1:02}.pt"))
    torch.save(torch.tensor(test_losses), os.path.join(project_dir, "test_losses", f"epoch_{epoch+1:02}.pt"))
    torch.save(autoencoder.state_dict(), os.path.join(project_dir, "weights", f"epoch_{epoch+1:02}.pt"))

    # Save best weights for model
    if best_test_loss is None or mean_test_loss < best_test_loss:
        best_test_loss = mean_test_loss
        torch.save(autoencoder.state_dict(), os.path.join(project_dir, "weights", f"best.pt"))

    # Save results
    results = pd.DataFrame(zip(accuracies, times), columns=("accuracy", "time"))
    results.to_csv(os.path.join(project_dir, "results.csv"))
