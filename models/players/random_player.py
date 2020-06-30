import random

from models.move import Move

class RandomPlayer:
	def __init__(self, player):
		self.player = player

	def get_board_move(self, game):
		return random.choice(game.valid_moves())

	def get_bid(self, game):
		return random.randint(0, game.coins[self.player])