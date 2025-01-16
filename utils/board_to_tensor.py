import chess
import numpy as np

# Mapping from piece type and color to tensor layer
PIECE_TO_LAYER = {
    chess.PAWN: 0,
    chess.ROOK: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board):
    """
    Convert a chess board to a tensor representation.

    Parameters
    ----------
    board : chess.Board
        The chess board to be converted.

    Returns
    -------
    numpy.ndarray
        A tensor of shape (8, 8, 12) representing the board. Each layer corresponds to a specific piece type and color.
        The first 6 layers represent white pieces (pawn, knight, bishop, rook, queen, king) and the next 6 layers represent black pieces.
        The tensor is initialized to zeros, and a value of 1 is set for the presence of a piece at the corresponding position.
    """
    # Create an 8x8x12 tensor initialized to zeros
    tensor = np.zeros((8, 8, 12), dtype=int)

    # Iterate over all 64 squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)  # Convert square index to row and column
            # each piece and its color will have its layer on the CNN
            # (e.g., white pawns = layer 0, black queen = layer 10)
            layer = PIECE_TO_LAYER[piece.piece_type] + (0 if piece.color == chess.BLACK else 6)
            tensor[row, col, layer] = 1  # Set the tensor value to 1 for this piece

    return tensor
