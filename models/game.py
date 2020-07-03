# -*- coding: utf-8 -*-
import copy

from models.move import Move

class Game:
    N, INITIAL_COINS, EMPTY = 3, 100, -1

    def __init__(self, board=None, coins=None):
        self.board = copy.deepcopy(board) if board is not None else [[Game.EMPTY]*Game.N for _ in range(Game.N)]
        self.coins = coins if coins is not None else [self.INITIAL_COINS, self.INITIAL_COINS]

    def play(self, move, player):
        """Make move in the board"""
        valid = (player == 0 or player == 1) and (0 <= move.x < Game.N and 0 <= move.y < Game.N)
        if not valid:
            raise Exception('Movement not valid')
        # make move
        self.board[move.x][move.y] = player

    def update_coins(self, player, bids):
        """Update players coins, taking from the winner and giving to the loser"""
        if not all([self.validate_bid(bids[player], player) for player in range(2)]):
            raise Exception('Bid not valid')
        for player_to_update in range(2):
            if player_to_update == player: self.coins[player_to_update] -= bids[player]
            else: self.coins[player_to_update] += bids[player]

    def game_finished(self):
        """
        Returns info about game ending (is_game_finished, winner_player).
        The possible returns are:
            . (True, 0): Player 0 won the game
            . (True, 1): Player 1 won the game
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

    def get_clone(self):
        """Get clone object to pass to agents so that they cannot change the original game"""
        return Game(copy.deepcopy(self.board), self.coins)

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

    def get_state_array(self):
        """Get single array with [*self.board.flatten(), *self.coins]"""
        return [
            *[item for row in self.board for item in row],
            *self.coins
        ]
