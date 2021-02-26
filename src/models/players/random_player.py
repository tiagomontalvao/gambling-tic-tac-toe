import random

from models.move import Move
from models.players.base_player import BasePlayer


class RandomPlayer(BasePlayer):
    def get_bid(self, game):
        return random.randint(0, game.coins[self.player])

    def get_board_move(self, game):
        return random.choice(game.valid_moves())
