from models.move import Move

class HumanPlayer:
	def __init__(self, player):
		self.player = player

	def play(self, valid_moves):
		while True:
			try:
				move = Move(input("Move: "), kind='human')
				if move in valid_moves: break
				else: print("Invalid move.")
			except:
				continue
		return move