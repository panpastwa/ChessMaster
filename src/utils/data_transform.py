import torch
from torch.nn.functional import one_hot

# TODO: fill docstrings


PIECES_ENCODING = {
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12
}
PIECES_ENCODING_REVERSE = dict((v, k) for k, v in PIECES_ENCODING.items())


def fen_to_board_matrix(fen_string: str) -> torch.Tensor:
    """


    Args:
        fen_string: String with chess position encoded in FEN format

    Returns:
        8x8 matrix containing current board state

    """

    board_tensor = torch.zeros((8, 8), dtype=torch.long)
    position_repr = fen_string.split(' ')
    board_repr = position_repr[0]
    board_rows = board_repr.split('/')

    for i, row in enumerate(board_rows):
        j = 0
        for c in row:
            if c.isdigit():
                j += int(c)
            else:
                v = PIECES_ENCODING[c]
                board_tensor[i, j] = v
                j += 1

    return board_tensor


def board_matrix_to_fen(board_tensor: torch.Tensor) -> str:
    """


    Args:
        board_tensor:

    Returns:

    """

    assert board_tensor.shape == torch.Size((8, 8))  # Ensure proper shape of tensor

    # Create FEN string from given matrix
    fen_string = ""
    counter = 0
    for row in board_tensor:
        for v in row:
            if v.item() in PIECES_ENCODING_REVERSE:
                if counter > 0:
                    fen_string += str(counter)
                    counter = 0
                fen_string += PIECES_ENCODING_REVERSE[v.item()]
            else:
                counter += 1
        if counter > 0:
            fen_string += str(counter)
            counter = 0
        fen_string += '/'

    fen_string = fen_string[:-1]  # Remove '/' at the end of string

    return fen_string


def fen_to_one_hot_matrix(fen_string: str) -> torch.Tensor:
    """


    Args:
        fen_string: String with chess position encoded in FEN format

    Returns:
        2D tensor of shape (8, 8, 13) representing chess position with pieces encoded as one-hot vector

    """

    board_matrix = fen_to_board_matrix(fen_string)
    one_hot_matrix = one_hot(board_matrix, num_classes=13)
    return one_hot_matrix


def one_hot_matrix_to_fen(board_tensor: torch.Tensor) -> str:
    """


    Args:
        board_tensor: Tensor of shape (8, 8, 13) containing bitboard position representation

    Returns:
        String with encoded position in FEN format

    """

    assert board_tensor.shape == torch.Size((8, 8, 13))  # Ensure proper shape of tensor
    indices = torch.nonzero(board_tensor)[:, -1].reshape((8, 8))  # Extract indices of pieces
    fen_string = board_matrix_to_fen(indices)
    return fen_string


def fen_to_full_position_tensor(fen_string: str) -> torch.Tensor:
    """


    Args:
        fen_string: String with chess position encoded in FEN format

    Returns:
        1D tensor containing bitboard representation of whole position (including castiling rights and side to move)

    """

    # 5 additional bits representing side to move and castling rights
    input_size = 64*13 + 5
    board_tensor = torch.zeros((input_size, ), dtype=torch.long)

    # Encode board position
    board_matrix = fen_to_one_hot_matrix(fen_string)
    board_tensor[:8*8*13] = torch.flatten(board_matrix)

    position_repr = fen_string.split(' ')
    assert len(position_repr) == 6
    side_to_move = position_repr[1]
    castle_rights = position_repr[2]

    # Encode side to move
    if side_to_move == 'w':
        board_tensor[-5] = 1

    # Encode castle rights
    if castle_rights != '-':
        for i, c in enumerate(['K', 'Q', 'k', 'q']):
            if c in castle_rights:
                board_tensor[-4+i] = 1

    return board_tensor


def full_position_tensor_to_fen(board_tensor: torch.Tensor) -> str:
    """


    Args:
        board_tensor:

    Returns:

    """
    assert board_tensor.shape == torch.Size((8*8*13 + 5, ))

    board_matrix = board_tensor[:8*8*13].reshape((8, 8, 13))
    fen_string = one_hot_matrix_to_fen(board_matrix)

    if board_tensor[-5]:
        fen_string += " w "
    else:
        fen_string += " b "

    for v, castle_right in zip(board_tensor[-4:], ('K', 'Q', 'k', 'q')):
        if v:
            fen_string += castle_right

    return fen_string
