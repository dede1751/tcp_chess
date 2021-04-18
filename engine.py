"""
This module implements the basic logic for the chess game.
It implements move legality checking and fen string translation.

    BOARD  -- Represented by np.ndarray of 8-bit ints with length 64. Board is
              indexed from the top left to the bottom right, so:
                A8, B8, ... , G1, H1  -->  board[0] will be a black rook

    PIECES -- Represented by 8-bit unsigned ints (0-256)

    2^7    |  2^6   |  2^5  |   2^4   |  2^3   |  2^2   |  2^1   | 2^0
 en-passant| color  | king  |  queen  |  rook  | bishop | knight | pawn

    e.g. 00100000 == 32 -> white king
         01001000 == 72 -> black rook

        *   piece & n checks if it's a piece worth n (eg n=16 checks for queen)
            piece >> 6 returns the color
        *   10000000 will be used for en passant "e" (en-passant squares also
            have a color bit to avoid hitting your own pawns)

    TURNS -- Represented by 1 bit: 1 == "b" and 0 == "w" to match color bit

    MOVES -- Represented either by lists or single ints, depending on whether
             the start position is known or not:

        *  (8, 16), assuming the starting board, is a6 by the black a-pawn.
           this same move can be represented by 16 if only looking at the
           black a-pawn (e.g. when checking if the pawn's move is legal).
        *  castling is simply treated as a 2-tile move by the king

    CASTLE -- 2x2 np.ndarray of 4 boolean values, following fen standard KQkq.
              First row is friendly, second is enemy, swapped each time the
              castle state is updated so that order is maintained through turns.

    Tried implementing it as a 4 bit binary value, but since python
    ints are dynamically allocated 0's get compressed.
    (1111 & 0101 & 1100 + 1111 & 0011, which equates to removing the
    white kingside castle, returns 0b0 in python)


-- Move generation:

For simplicity, I'll resort to pseudo-legal move generation and simply check the
opponent's moves for any checking move.
After I finish this and clean up the rest of the code for the game, I'll work on
this properly by moving over to bitboards and write engine code in C for easier
bit manipulation.

https://peterellisjones.com/posts/generating-legal-chess-moves-efficiently/
"""

import numpy as np

# basic types
Board = np.ndarray # 64 x 1 array
Move = list[int,int]
Castle = np.ndarray # 2 x 2 array
Match = list[Board, Castle, int] # board, castle, turn

# PRECOMPUTED STATIC VALUES/TABLES
start_white = np.array([
    72, 66, 68, 80, 96, 68, 66, 72,
    65, 65, 65, 65, 65, 65, 65, 65,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     1,  1,  1,  1,  1,  1,  1,  1,
     8,  2,  4, 16, 32,  4,  2,  8,
    ], dtype=np.uint8)
start_black = np.array([
     8,  2,  4, 32, 16,  4,  2,  8,
     1,  1,  1,  1,  1,  1,  1,  1,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,
    65, 65, 65, 65, 65, 65, 65, 65,
    72, 66, 68, 96, 80, 68, 66, 72,
    ], dtype = np.uint8)
masks = {
    "k": 0b100000, "q": 0b010000, "r": 0b001000,
    "b": 0b000100, "n": 0b000010, "p": 0b000001,
    "e": 0b10000000,
    }
offsets = [-8, 8, -1, 1, -9, -7, 7, 9] # n, s, w, e, nw, ne, sw, se
distances = np.array([
    [0, 7, 0, 7, 0, 0, 0, 7], [0, 7, 1, 6, 0, 0, 1, 6], [0, 7, 2, 5, 0, 0, 2, 5],
    [0, 7, 3, 4, 0, 0, 3, 4], [0, 7, 4, 3, 0, 0, 4, 3], [0, 7, 5, 2, 0, 0, 5, 2],
    [0, 7, 6, 1, 0, 0, 6, 1], [0, 7, 7, 0, 0, 0, 7, 0], [1, 6, 0, 7, 0, 1, 0, 6],
    [1, 6, 1, 6, 1, 1, 1, 6], [1, 6, 2, 5, 1, 1, 2, 5], [1, 6, 3, 4, 1, 1, 3, 4],
    [1, 6, 4, 3, 1, 1, 4, 3], [1, 6, 5, 2, 1, 1, 5, 2], [1, 6, 6, 1, 1, 1, 6, 1],
    [1, 6, 7, 0, 1, 0, 6, 0], [2, 5, 0, 7, 0, 2, 0, 5], [2, 5, 1, 6, 1, 2, 1, 5],
    [2, 5, 2, 5, 2, 2, 2, 5], [2, 5, 3, 4, 2, 2, 3, 4], [2, 5, 4, 3, 2, 2, 4, 3],
    [2, 5, 5, 2, 2, 2, 5, 2], [2, 5, 6, 1, 2, 1, 5, 1], [2, 5, 7, 0, 2, 0, 5, 0],
    [3, 4, 0, 7, 0, 3, 0, 4], [3, 4, 1, 6, 1, 3, 1, 4], [3, 4, 2, 5, 2, 3, 2, 4],
    [3, 4, 3, 4, 3, 3, 3, 4], [3, 4, 4, 3, 3, 3, 4, 3], [3, 4, 5, 2, 3, 2, 4, 2],
    [3, 4, 6, 1, 3, 1, 4, 1], [3, 4, 7, 0, 3, 0, 4, 0], [4, 3, 0, 7, 0, 4, 0, 3],
    [4, 3, 1, 6, 1, 4, 1, 3], [4, 3, 2, 5, 2, 4, 2, 3], [4, 3, 3, 4, 3, 4, 3, 3],
    [4, 3, 4, 3, 4, 3, 3, 3], [4, 3, 5, 2, 4, 2, 3, 2], [4, 3, 6, 1, 4, 1, 3, 1],
    [4, 3, 7, 0, 4, 0, 3, 0], [5, 2, 0, 7, 0, 5, 0, 2], [5, 2, 1, 6, 1, 5, 1, 2],
    [5, 2, 2, 5, 2, 5, 2, 2], [5, 2, 3, 4, 3, 4, 2, 2], [5, 2, 4, 3, 4, 3, 2, 2],
    [5, 2, 5, 2, 5, 2, 2, 2], [5, 2, 6, 1, 5, 1, 2, 1], [5, 2, 7, 0, 5, 0, 2, 0],
    [6, 1, 0, 7, 0, 6, 0, 1], [6, 1, 1, 6, 1, 6, 1, 1], [6, 1, 2, 5, 2, 5, 1, 1],
    [6, 1, 3, 4, 3, 4, 1, 1], [6, 1, 4, 3, 4, 3, 1, 1], [6, 1, 5, 2, 5, 2, 1, 1],
    [6, 1, 6, 1, 6, 1, 1, 1], [6, 1, 7, 0, 6, 0, 1, 0], [7, 0, 0, 7, 0, 7, 0, 0],
    [7, 0, 1, 6, 1, 6, 0, 0], [7, 0, 2, 5, 2, 5, 0, 0], [7, 0, 3, 4, 3, 4, 0, 0],
    [7, 0, 4, 3, 4, 3, 0, 0], [7, 0, 5, 2, 5, 2, 0, 0], [7, 0, 6, 1, 6, 1, 0, 0],
    [7, 0, 7, 0, 7, 0, 0, 0],
    ], dtype=np.uint8)
knight_moves = [
    [10, 17], [10, 17, 15], [10, 17, 15, 6], [10, 17, 15, 6], [10, 17, 15, 6],
    [10, 17, 15, 6], [17, 15, 6], [15, 6], [-6, 10, 17], [-6, 10, 17, 15],
    [-6, 10, 17, 15, 6, -10], [-6, 10, 17, 15, 6, -10], [-6, 10, 17, 15, 6, -10],
    [-6, 10, 17, 15, 6, -10], [17, 15, 6, -10], [15, 6, -10], [-15, -6, 10, 17],
    [-15, -6, 10, 17, 15, -17], [-15, -6, 10, 17, 15, 6, -10, -17],
    [-15, -6, 10, 17, 15, 6, -10, -17], [-15, -6, 10, 17, 15, 6, -10, -17],
    [-15, -6, 10, 17, 15, 6, -10, -17], [-15, 17, 15, 6, -10, -17],
    [15, 6, -10, -17], [-15, -6, 10, 17], [-15, -6, 10, 17, 15, -17],
    [-15, -6, 10, 17, 15, 6, -10, -17], [-15, -6, 10, 17, 15, 6, -10, -17],
    [-15, -6, 10, 17, 15, 6, -10, -17], [-15, -6, 10, 17, 15, 6, -10, -17],
    [-15, 17, 15, 6, -10, -17], [15, 6, -10, -17], [-15, -6, 10, 17],
    [-15, -6, 10, 17, 15, -17], [-15, -6, 10, 17, 15, 6, -10, -17],
    [-15, -6, 10, 17, 15, 6, -10, -17], [-15, -6, 10, 17, 15, 6, -10, -17],
    [-15, -6, 10, 17, 15, 6, -10, -17], [-15, 17, 15, 6, -10, -17],
    [15, 6, -10, -17], [-15, -6, 10, 17], [-15, -6, 10, 17, 15, -17],
    [-15, -6, 10, 17, 15, 6, -10, -17], [-15, -6, 10, 17, 15, 6, -10, -17],
    [-15, -6, 10, 17, 15, 6, -10, -17], [-15, -6, 10, 17, 15, 6, -10, -17],
    [-15, 17, 15, 6, -10, -17], [15, 6, -10, -17], [-15, -6, 10],
    [-15, -6, 10, -17], [-15, -6, 10, 6, -10, -17], [-15, -6, 10, 6, -10, -17],
    [-15, -6, 10, 6, -10, -17], [-15, -6, 10, 6, -10, -17], [-15, 6, -10, -17],
    [6, -10, -17], [-15, -6], [-15, -6, -17], [-15, -6, -10, -17],
    [-15, -6, -10, -17], [-15, -6, -10, -17], [-15, -6, -10, -17],
    [-15, -10, -17]
    ]

def flip_board(board: Board) -> None:
    for tile in range(32):
        flip = 63 - 8*(tile // 8) - (tile % 8)
        board[tile], board[flip] = board[flip], board[tile]

def is_capture(piece: int, turn: int): # true if enemy, else false
    return piece != 0 and (piece >> 6 != turn)

def tile_available(piece: int, turn: int): # false if friendly, else true
    return piece == 0 or (piece >> 6 != turn) or (piece & masks["e"])


def update_castle_status(game: Match, move: Move):
    """
    Modifies the castling array based on the move made, checks:
        - if and which rook moves or is captured
        - if the king moves
    """
    board, castle, turn = game
    start, end = move

    if is_capture(board[end], turn) and board[end] & masks["r"]: # enemy rook
        if end % 8 == 0:
            castle[1, 1] = 0
        elif start % 8 == 7:
            castle[1, 0] = 0

    if board[start] & masks["r"]: # friendly rook
        if start % 8 == 0:
            castle[0, 1] = 0
        elif start % 8 == 7:
            castle[0, 0] = 0
    elif board[start] & masks["k"]: #friendly king
        castle[0, 0] = 0
        castle[0, 1] = 0

    castle[[0,1]] = castle[[1,0]] # swap friendly/enemy castle


def execute_move(game: Match, move: Move) -> Board:
    """
    Creates new board array on which the move is executed.
    It also handles moving the rook when castling and setting/removing
    en-passant flags. Promotions are for now limited to queens.
    """
    board, _, turn = game # don't need castle info
    start, end = move
    board = np.copy(board)

    if board[start] & masks["k"]: #handle castling
        length = end - start
        if length == -2: # FIX CASTLING
            board[end + 1], board[end - 2] = board[end - 2], 0
        elif length == 2:
            board[end - 1], board[end + 1] = board[end + 1], 0

    elif board[start] & masks["p"]: # handle en-passant/promotions
        if end - start == -16: # double move
            board[start - 8] = 128 + 64*turn
        elif board[end] & masks["e"]: # en passant capture
            board[end + 8] = 0
        elif end // 8 == 0: # promotion (defaulted to queen)
            board[end] = 16 + 64*turn

    for tile in range(16, 24): # clean up old en passants
        if board[tile] & masks["e"]:
            board[tile] = 0
            break

    board[end] = board[start]
    board[start] = 0

    return board


def check_legality(game: Match, move: Move):
    """
    Verifies if a given move is legal by going through the pseudo-legal moves
    of the gien piece and then looking for checks in the resulting position.
    """
    board, _, turn = game
    start, end = move
    piece = board[start]

    if piece & masks["p"]:
        mvs = get_pawn_moves(board, turn, start)
        cap = get_pawn_captures(board, turn, start)
        moves = mvs + cap
    elif piece & masks["n"]:
        moves = get_knight_moves(board, turn, start)
    else:
        moves = get_slide_moves(board, turn, start) # king included

    if piece & masks["k"]:
        moves += get_castling_moves(game, start)

    if end in moves and not king_in_check(execute_move(game, move), turn):
        return True
    else:
        return False


def get_pawn_moves(board: Board, turn: int, start: int) -> list[int]:
    moves = []
    if board[start - 8] == 0:
        moves.append(start - 8)
        if board[start - 16] == 0 and (start // 8) == 6:
            moves.append(start - 16)

    return moves


def get_pawn_captures(board: Board, turn: int, start: int) -> list[int]:
    moves = []
    captures = [-7, -9]
    for cap in captures:
        move = start + cap
        if is_capture(board[move], turn): # is_capture includes en passant
            moves.append(move)

    return moves


def get_knight_moves(board: Board, turn: int, start: int) -> list[int]:
    moves = []
    for step in knight_moves[start]:
        move = start + step
        if tile_available(board[move], turn):
            moves.append(move)

    return moves


def get_slide_moves(board: Board, turn: int, start: int) -> list[int]:
    moves = []
    piece = board[start]
    if piece & masks["k"]: # if block chooses distance/offset lists
        steps = offsets
        dists = [(1 if n != 0 else 0) for n in distances[start]]
    elif piece & masks["q"]:
        steps = offsets
        dists = distances[start]
    elif piece & masks["r"]:
        steps = offsets[:4]
        dists = distances[start][:4]
    elif piece & masks["b"]:
        steps = offsets[4:]
        dists = distances[start][4:]

    for step, dist in zip(steps, dists):
        for num in range(1, dist + 1):
            move = start + step * num
            if tile_available(board[move], turn):
                moves.append(move)
            if board[move] != 0 and (not board[move] & masks["e"]):
                break

    return moves


def get_king_tile(board: Board, turn: int) -> int:
    for start in range(64):
        if board[start] & masks["k"] and board[start] >> 6 == turn:
            return start

def king_in_check(board: Board, turn: int):
    """
    Based on the simple concept that, if your king is in check, you can find
    checking moves by swapping your king with all possible pieces and seeing if
    it can capture that same type of enemy piece.
    """
    start = get_king_tile(board, turn)

    for p in ["q", "b", "r", "k"]:
        val = masks[p]
        if turn == 1:
            val += 64
        board[start] = val
        for move in get_slide_moves(board, turn, start):
            if board[move] & masks[p] and board[move] >> 6 != turn:
                return True

    for move in get_knight_moves(board, turn, start):
        if board[move] & masks["n"] and board[move] >> 6 != turn:
            return True

    for move in get_pawn_captures(board, turn, start):
        if board[move] & masks["p"] and board[move] >> 6 != turn:
            return True

    return False


def get_castling_moves(game: Match, start: int) -> list[int]:
    moves = []
    board, castle, turn = game

    kingside, queenside = castle[0]

    if kingside and not (board[start + 1] or board[start + 2]):
        if not king_in_check(execute_move(game, (start, start + 1)), turn):
            moves.append(start + 2)
    if queenside and not (board[start - 1] or board[start - 2] or board[start - 3]):
        if not king_in_check(execute_move(game, (start, start - 1)), turn):
            moves.append(start - 2)

    return moves


def update_game(game: Match, move: Move) -> None:
    update_castle_status(game, move)
    game[0] = execute_move(game, move)
    game[2] = not game[2]
