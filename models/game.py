# -*- coding: utf-8 -*-
import copy

from models.move import Move
from views.console_game_view import ConsoleGameView

class Game:
    N, INITIAL_COINS, EMPTY, *PLAYERS = 3, 100, 0, 0, 1, 2

    def __init__(self, board=None, coins_p1=None, coins_p2=None, player=None):
        self.board = copy.deepcopy(board) if board is not None else [[Game.EMPTY]*Game.N for _ in range(Game.N)]
        self.coins_p1 = coins_p1 if coins_p1 is not None else self.INITIAL_COINS
        self.coins_p2 = coins_p2 if coins_p2 is not None else self.INITIAL_COINS
        self.player = player if player is not None else 1
        self.view = ConsoleGameView(self.board, coins_p1, coins_p2, player)

    def play(self, move: Move, player: int):
        """Make move in the board"""
        valid = (player == Game.PLAYERS[1] or player == Game.PLAYERS[2]) and (0 <= move.x < Game.N and 0 <= move.y < Game.N)
        if not valid:
            raise Exception('Movement not valid')
        # make move
        self.board[move.x][move.y] = player

    def game_finished(self):
        """
        Returns info about game ending (is_game_finished, winner_player).
        The possible returns are:
            . (True, Game.PLAYER_1): Player 1 won the game
            . (True, Game.PLAYER_2): Player 2 won the game
            . (True, None): The game endeded in a tie
            . (False, None): Game is not over yet
        """
        for i in range(Game.N):
            if len(set([self.board[i][j] for j in range(Game.N)])) == 1 and self.board[i][0] != Game.EMPTY:
                return True, self.board[i][0]
            if len(set([self.board[j][i] for j in range(Game.N)])) == 1 and self.board[0][i] != Game.EMPTY:
                return True, self.board[0][i]
        if len(set([self.board[i][i] for i in range(Game.N)])) == 1 and self.board[0][0] != Game.EMPTY:
            return True, self.board[0][0]
        if len(set([self.board[i][Game.N-1-i] for i in range(Game.N)])) == 1 and self.board[0][Game.N-1] != Game.EMPTY:
            return True, self.board[0][Game.N-1]
        if len([self.board[i][j] for i in range(Game.N) for j in range(Game.N) if self.board[i][j] == Game.EMPTY]) == 0:
            return True, Game.EMPTY
        return False, None

    def _opponent(self, player: int):
        """Returns opponent player of `player`"""
        return Game.PLAYERS[1] if player is Game.PLAYERS[2] else Game.PLAYERS[2]