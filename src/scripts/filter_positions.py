import sys

import pandas as pd


def filter_positions(input_csv, output_csv):

    chunksize = 10 ** 6
    header = True
    columns = ["white_elo", "black_elo", "time_control", "move_ply", "termination",
               "white_won", "black_won", "no_winner", "cp", "board"]
    with pd.read_csv(input_csv, chunksize=chunksize, usecols=columns) as reader:
        for chunk in reader:

            # Filter by ELO
            chunk = chunk[chunk["white_elo"] > 2000]
            chunk = chunk[chunk["black_elo"] > 2000]

            # Filter by time_control
            mask = chunk["time_control"].str.split('+').str.get(0).str.len() > 2
            chunk = chunk[mask]

            # Filter by move number
            chunk = chunk[chunk["move_ply"] > 24]

            # Take only subset of positions from games
            chunk = chunk.sample(frac=1/10)

            chunk.to_csv(output_csv, header=header, index=False, mode='a')
            header = False


if __name__ == "__main__":
    if len(sys.argv) == 3:
        filter_positions(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python3 filter_positions.py INPUT_CSV OUTPUT_CSV")
