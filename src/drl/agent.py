import numpy as np
import random
from collections import namedtuple, deque

from drl.ddpg_agent import DDPGAgent

import torch
import torch.nn.functional as F

class DRLAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, bid_action_size, board_action_size, seed=None, checkpoint_path=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            bid_action_size (int): Dimension of each bid action
            board_action_size (int): Dimension of each board action (number of cells in the game table)
            seed (int): Random seed of PRNG engines
            checkpoint_path (string): Directory with saved model checkpoints
        """
        self.agent = DDPGAgent(state_size, bid_action_size, board_action_size, seed, checkpoint_path)

    def act(self, state):
        return self.agent.act(state)

        # state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)

        # self.network.eval()
        # with torch.no_grad():
        #     bid, board_move, value = self.network(state)
        # self.network.train()

        # bid = int(bid)
        # board_move = torch.max(board_move, dim=1)[1]

        # return bid, board_move, value

    # def step(self, state, action_bid, action_board_move, reward, next_state, done):
    def step(self, state, action, reward, next_state, done):
        return self.agent.step(state, action, reward, next_state, done)
