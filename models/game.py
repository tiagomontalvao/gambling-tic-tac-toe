# -*- coding: utf-8 -*-
import copy

from models.move import Move
from views.console_game_view import ConsoleGameView

class Game:
    N, INITIAL_COINS, EMPTY, *PLAYERS = 3, 100, ' ', *ConsoleGameView.CELL_CHAR

    def __init__(self, board=None, coins=None, player=None):
        self.board = copy.deepcopy(board) if board is not None else [[Game.EMPTY]*Game.N for _ in range(Game.N)]
        if coins is not None and len(coins) == 2: coins = [None] + coins
        self.coins = coins if coins is not None else [None, self.INITIAL_COINS, self.INITIAL_COINS]
        self.player = player if player is not None else 1

    def play(self, move, player, bids):
        """Make move in the board"""
        valid = (player == 1 or player == 2) and (0 <= move.x < Game.N and 0 <= move.y < Game.N)
        if not valid:
            raise Exception('Movement not valid')
        if not all([self.validate_bid(bids[player], player) for player in range(1, 3)]):
            raise Exception('Bid not valid')
        # make move
        self.board[move.x][move.y] = self.PLAYERS[player]
        # update coins
        print('before', self.coins)
        for player_to_update in range(1, 3):
            if player_to_update == player: self.coins[player_to_update] -= bids[player]
            else: self.coins[player_to_update] += bids[player]
        print('after', self.coins)

    def game_finished(self):
        """
        Returns info about game ending (is_game_finished, winner_player).
        The possible returns are:
            . (True, Game.PLAYERS[1]): Player 1 won the game
            . (True, Game.PLAYERS[2]): Player 2 won the game
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

    def valid_moves(self):
        """Return all empty cells"""
        ret = []
        for i in range(Game.N):
            for j in range(Game.N):
                if self.board[i][j] == Game.EMPTY:
                    ret += [Move(i, j)]
        return ret

    def validate_bid(self, bid, player):
        """Validates if bid made by player is valid, i.e., is in the range [0, coins[player]]"""
        return 0 <= bid <= self.coins[player]