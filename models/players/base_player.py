from abc import ABC, abstractmethod

class BasePlayer(ABC):
    """Abstract base player that must be extended by all implemented players"""
    def __init__(self, player):
        """BasePlayer constructor
        Params
        ======
            player (int): id assigned to this player
        """
        self.player = player

    @abstractmethod
    def get_bid(self, game):
        """Get player bid so that the winning bid gives the player the right to move in the board
        Params
        ======
            game (models.Game): game object to give info to player of what move to make
        """
        pass

    @abstractmethod
    def get_board_move(self, game):
        """Get move in board after placing winning bid
        Params
        ======
            game (models.Game): game object to give info to player of what move to make
        """
        pass

    def sinalize_done(self, winner):
        """Sinalize that game ended and who is the winner
        Params
        ======
            winner (int): id of winning player
        """
        pass