"""
This module implements solely the gui elements of the chess game. It does not
care for efficiency, implements no logic and simply allows the player to see
and interact with the board.

The board is a 1D vector, indexed from the top left to bottom right
(A8,B8, ... , G1, H1). For more info check engine module.
"""

import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # avoid pygame prompt

import pygame as pg
from pygame.sprite import Sprite, Group
import numpy as np
import engine


class Point(Sprite):
    """Pygame trick for mouse interaction with sprites"""
    def __init__(self, pos: tuple[int,int]) -> None:
        super().__init__()
        self.rect = pg.Rect(pos[0], pos[1], 1, 1)
        self.rect.center = pos


class ChessPiece(Sprite):
    """
    Chess piece sprite with simple tile movement.
        - ChessPiece.piece         8-bit integer representing the piece
        - ChessPiece.color         1-bit representing color
        - ChessPiece.last_tile     memory for replacing piece at original pos
    """
    def __init__(self, pos: int, piece: str) -> None:
        super().__init__()
        self.piece = piece
        self.color = piece >> 6

        self.image = pg.image.load(f"images/{piece}.png")
        self.rect = self.image.get_rect()

        self.move_to_tile(pos)

    def move_to_tile(self, move: int) -> None:
        self.rect.center = (move % 8)*62.5 + 31.25, (move // 8)*62.5 + 31.25
        self.last_tile = move


class PieceGroup(Group):
    """Sprite group wrapper, avoids passing game instance to all sprites"""
    def __init__(self, game: 'ChessGame') -> None:
        super().__init__()
        self.screen = game.screen

    def update_sprites(self) -> None:
        if game.grab: # grabbed piece follows mouse
            game.grab.rect.center = pg.mouse.get_pos()
        for p in super().sprites():
            self.screen.blit(p.image, p.rect)


class ChessGame():
    """
    Game instance is controlled by main script, hence not much is done at init.
        - ChessGame.active indicates whether the game instance is running the
          gui.
        - start_new_game(side) does what would normally be at init
        - setup_board() updates the game to the current match attribute
          everything regarding move checking and board generation should be
          handled by the engine module, and then the result submitted to this
          class by updating said match attribute.
    """
    def __init__(self) -> None:
        self.active = False
        self.point = Point([0, 0])

    def start_new_game(self, side: int) -> None:
        pg.init()
        self.image = pg.image.load("images/board.png")
        self.rect = self.image.get_rect()
        self.screen = pg.display.set_mode((500, 500))
        pg.display.set_caption("Chess")

        self.side = side        # side to display board from
        self.grab = None        # currently grabbed piece
        self.player_move = None # (start, end) scalar coordinates
        self.active = True      # pygame window activity

        if side == 0:
            start = np.copy(engine.start_white)
        else:
            start = np.copy(engine.start_black)
        self.match = [start, np.ones((2,2), dtype=np.uint8), 0]
        self.setup_board()

    def setup_board(self) -> None:
        """Spawns and places pieces based on board input."""
        self.turn = self.match[2]
        self.pieces = PieceGroup(self)

        for tile in range(64):
            piece = self.match[0][tile]
            if piece and not (piece & 0b10000000): # don't spawn en passant
                self.pieces.add(ChessPiece(tile, piece))

    def close_window(self) -> None: # close window without killing class
        pg.display.quit()
        pg.quit()
        self.active = False

    def check_events(self) -> None:
        """Event loop. Press Q or exit window to quit current instance"""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close_window()
            elif event.type == pg.KEYDOWN   :
                if event.key == pg.K_q:
                    self.close_window()
            elif event.type == pg.MOUSEBUTTONDOWN:
                self.grab_piece()
            elif event.type == pg.MOUSEBUTTONUP and self.grab:
                self.drop_piece()

    def grab_piece(self) -> None:
        """Checks for sprite collisions."""
        self.point.rect.center = pg.mouse.get_pos()
        p = pg.sprite.spritecollide(self.point, self.pieces, False)
        if p: # in case you don't click any piece
            piece = p[0]
            self.grab = piece

    def drop_piece(self) -> None:
        """Translates player movement to engine-readable move"""
        # grabbed piece can't be moved (opponent's piece or opponent's turn)
        if not (self.grab.color == self.turn and self.turn == self.side):
            self.grab.move_to_tile(self.grab.last_tile) # reset
            self.grab = None
            return

        start = self.grab.last_tile
        x, y = pg.mouse.get_pos()
        end = (int(y // 62.5))*8 + int(x // 62.5)
        move = [start, end]

        if start != end and engine.check_legality(self.match, move):
            engine.update_game(self.match, move)
            self.setup_board()
            self.player_move = move # start broadcasting new move
        else:
            self.grab.move_to_tile(self.grab.last_tile)

        self.grab = None

    def display_frame(self) -> None:
        """Checks events, blits to surfaces and updates display"""
        self.check_events()
        if self.active: # to avoid errors when closing instance
            self.screen.blit(self.image, self.rect)
            self.pieces.update_sprites()
            pg.display.flip()
        pg.time.wait(3)
