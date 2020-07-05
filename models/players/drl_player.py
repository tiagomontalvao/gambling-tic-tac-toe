import numpy as np

from drl.agent import DRLAgent
from models.move import Move
from models.players.base_player import BasePlayer

class DRLPlayer(BasePlayer):
    def __init__(self, player, game, train_mode=False):
        super(DRLPlayer, self).__init__(player)

        # agent_weights='../../drl/weights/model.pt'
        self.player = player
        self.agent = DRLAgent(state_size=3*3+2, board_size=3*3)
        self.train_mode = train_mode

        self.last_state = None
        self.last_bid = None
        self.last_board_move = None

    def get_bid(self, game):
        """Update agent's state, make action, save new state and return bid"""
        curr_state = game.get_state_array()

        self._get_experience_tuple_then_step(reward=0, done=False)

        # get action
        bid, board_move, _ = self.agent.act(curr_state)

        self.last_state = curr_state
        self.last_bid = np.clip(bid, 0, game.coins[self.player])
        self.last_board_move = board_move

        return bid

    def get_board_move(self, game):
        """
        Return move already saved in self.board_move.
        It is saved with a number from 0 to 9 and is converted to Move(x, y).
        """
        x = self.last_board_move // 3
        y = self.last_board_move % 3
        return Move(x, y)

    def sinalize_done(self, winner):
        """Perform step in the agent if self.train_mode=True after game is finished"""
        reward = 1 if winner == self.player else -1
        self._get_experience_tuple_then_step(reward=reward, done=True)

    def _get_experience_tuple_then_step(self, reward, done):
        """Perform step in the agent if self.train_mode=True"""
        if self.train_mode and self.last_state is not None:
            # create tuple (s, a, r, s', done)
            experience_tuple = (self.last_state, self.last_bid, self.last_board_move, reward, curr_state, done)
            # add tuple to agent
            self.agent.step(self.last_state, self.last_bid, self.last_board_move, reward, curr_state, done)
