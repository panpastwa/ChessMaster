import torch
import chess.pgn
from src.utils.data_transform import fen_to_one_hot_matrix


def pgn_extract_embeddings(autoencoder, pgn: str, device='cpu') -> torch.Tensor:
    """
    Extracts embeddings for given PGN file

    Args:
        autoencoder: Autoencoder model with proper weights
        pgn: Path to PGN file
        device: Device to run inference

    Returns:
        torch.Tensor of shape [N_POSITIONS, EMBEDDING_SIZE]

    """

    fens = []
    with open(pgn) as mf:
        game = chess.pgn.read_game(mf)
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            fens.append(board.fen())

    return fens_extract_embeddings(autoencoder, fens, device)


def fens_extract_embeddings(autoencoder, fens: list, device='cpu'):
    """
    Extracts embeddings for given FEN positions

    Args:
        autoencoder: Autoencoder model with proper weights
        fens: List of positions in FEN format
        device: Device to run inference

    Returns:
        torch.Tensor of shape [N_POSITIONS, EMBEDDING_SIZE]

    """

    with torch.no_grad():
        positions = torch.tensor([fen_to_one_hot_matrix(fen).tolist() for fen in fens])
        x = positions.flatten(start_dim=1).float().to(device)
        emb = autoencoder.encoder(x)
        return emb
