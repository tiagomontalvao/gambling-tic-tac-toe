import random
from collections import deque

from models.move import Move
from models.players.base_player import BasePlayer


class SemiRandomPlayer(BasePlayer):
    """Agent that plays completely random until it is able to win in a row.
    For example: if 3 spots in the board are empty in a row and the player has at least 176 coins,
        it can choose each one of these spots with the bids [25, 50, 100].
    This agent assumes the board is 3x3 and that the total coins of both players is 200.
    """

    def __init__(self, player, game):
        super().__init__(player, game)

        assert game.N == 3

        self._reset_state()
        self._check_functions = [
            self._check_row,
            self._check_col,
            self._check_diag
        ]

    def get_bid(self, game):
        self._try_to_win_in_sequence(game)

        if self.next_plays is not None:
            self.bid, self.board_move = self.next_plays.popleft()
            return self.bid

        return random.randint(0, game.coins[self.player])

    def get_board_move(self, game):
        # If move is already pre-computed, just return it
        if self.next_plays is not None:
            return self.board_move

        return random.choice(game.valid_moves())

    def sinalize_done(self, winner):
        self._reset_state()

    def _reset_state(self):
        """Reset move variables"""
        self.next_plays = None
        self.bid = None
        self.board_move = None

    def _try_to_win_in_sequence(self, game):
        """
        If the game can be won in a few moves based on the board and coins of each player, win it.
        For example, if the player has more than 
        """
        if self.next_plays is not None:
            return

        missing_spots_coins = {
            1: 101,
            2: 151,
            3: 176
        }

        random.shuffle(self._check_functions)
        for check_function in self._check_functions:
            if check_function(game, missing_spots_coins):
                return

    def _check_row(self, game, missing_spots_coins):
        """Check if a row can be won with movements in a row"""
        for row in random.sample(range(3), 3):
            # there is already a move for the opponent in this row
            if any(game.board[row][col] == (1-self.player) for col in range(3)):
                continue

            n_empty_spots = sum(game.board[row][col] ==
                                game.EMPTY for col in range(3))
            if game.coins[self.player] >= missing_spots_coins[n_empty_spots]:
                moves = [(row, col)
                         for col in range(3) if game.board[row][col] == game.EMPTY]
                return self._populate_next_plays(moves, n_empty_spots)

    def _check_col(self, game, missing_spots_coins):
        """Check if a column can be won with movements in a row"""
        for col in random.sample(range(3), 3):
            # there is already a move for the opponent in this col
            if any(game.board[row][col] == (1-self.player) for row in range(3)):
                continue

            n_empty_spots = sum(game.board[row][col] ==
                                game.EMPTY for row in range(3))
            if game.coins[self.player] >= missing_spots_coins[n_empty_spots]:
                moves = [(row, col)
                         for row in range(3) if game.board[row][col] == game.EMPTY]
                return self._populate_next_plays(moves, n_empty_spots)

    def _check_diag(self, game, missing_spots_coins):
        """Check if a diagonal can be won with movements in a row"""
        diagonals = [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]
        for diagonal in random.sample(diagonals, 2):
            # there is already a move for the opponent in this diagonal
            if any(game.board[row][col] == (1-self.player) for row, col in diagonal):
                continue

            n_empty_spots = sum(game.board[row][col] ==
                                game.EMPTY for row, col in diagonal)
            if game.coins[self.player] >= missing_spots_coins[n_empty_spots]:
                moves = [(row, col)
                         for row, col in diagonal if game.board[row][col] == game.EMPTY]
                return self._populate_next_plays(moves, n_empty_spots)

    def _populate_next_plays(self, moves, n_empty_spots):
        """Create next_plays deque based on """
        missing_spots_bids = {
            1: [100],
            2: [50, 100],
            3: [25, 50, 100],
        }

        self.next_plays = deque((bid, Move(*move))
                                for bid, move in zip(missing_spots_bids[n_empty_spots], moves))

        return True
