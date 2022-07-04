import os
import sys
import logging
import time

import torch
import pandas as pd

from src.models.SimpleAutoEncoder import SimpleAutoEncoder
from src.configs.autoencoder_800 import *
from src.train import train
from src.validate import validate


def run(dataset_dir):

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create new run dir
    if "runs" not in os.listdir("."):
        os.mkdir("runs")
        project_dir = os.path.join("runs", "001")
    else:
        last_dir = max(os.listdir("runs"))
        project_dir = os.path.join("runs", f"{int(last_dir) + 1:03}")
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
                 f"HIDDEN SIZE: {HIDDEN_SIZE}\n"  # For SimpleAutoEncoder
                 # f"LAYER SIZES: {LAYER_SIZES}\n"  # For ExtendedAutoEncoder
                 f"EPOCHS: {EPOCHS}\n"
                 f"LEARNING RATE: {LEARNING_RATE}\n")

    # Prepare data
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    train_files = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]
    test_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir)]
    positions_per_file = torch.load(train_files[0]).shape[0]
    logging.info(f"{len(train_files)} files will be used for training and {len(test_files)} will be used for testing. "
                 f"Each file contain {positions_per_file} postions.\n")

    # Create model, loss function and optimizer
    autoencoder = SimpleAutoEncoder(64 * 13, HIDDEN_SIZE).to(DEVICE)
    # autoencoder = ExtendedAutoEncoder(LAYER_SIZES).to(DEVICE)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(autoencoder.parameters(), lr=LEARNING_RATE, momentum=0.9)

    times, accuracies, pieces_accuracies, test_losses = [], [], [], []
    best_test_loss = None

    for epoch in range(EPOCHS):

        # Train and validate
        start_time = time.time()
        train_losses = train(autoencoder, train_files, BATCH_SIZE, DEVICE,
                             optimizer, loss_function, positions_per_file)
        test_loss, accuracy, pieces_accuracy = validate(autoencoder, test_files, BATCH_SIZE, DEVICE,
                                                        loss_function, positions_per_file)
        end_time = time.time()

        # Accumulate results
        times.append(end_time - start_time)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        pieces_accuracies.append(pieces_accuracy)

        # Save weights, results and train losses
        torch.save(autoencoder.state_dict(), os.path.join(project_dir, "weights", f"epoch_{epoch+1:02}.pt"))
        pd.DataFrame(data=zip(test_losses, accuracies, pieces_accuracies, times),
                     columns=("test_loss", "accuracy", "pieces_accuracy", "time")).\
            to_csv(os.path.join(project_dir, "results.csv"))
        pd.DataFrame(train_losses, columns=("train_loss", )).\
            to_csv(os.path.join(project_dir, "train_losses", f"epoch_{epoch+1:02}.csv"))

        # Save best weights for model
        if best_test_loss is None or test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(autoencoder.state_dict(), os.path.join(project_dir, "weights", f"best.pt"))

        logging.info(f"Epoch: {epoch + 1:3d} | "
                     f"Training loss: {torch.tensor(train_losses).mean():.3f} | "
                     f"Test loss: {test_loss:.3f} | "
                     f"Accuracy: {accuracy:.3f} | "
                     f"Pieces accuracy: {pieces_accuracy:.3f}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        print("Usage: python3 experiments.py dataset_dir")
