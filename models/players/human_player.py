from models.move import Move

class HumanPlayer:
	def __init__(self, player):
		self.player = player

	def get_board_move(self, valid_moves):
		while True:
			try:
				move = Move(input("Move: "), kind='human')
				if move in valid_moves: break
				else: print("Invalid move.")
			except ValueError:
				continue
		return move

	def get_bid(self):
		while True:
			try: bid = int(input())
			except ValueError:
				print('Invalid input. Try again: ', end='')
				continue
			return bid