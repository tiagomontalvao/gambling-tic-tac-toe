import numpy as np
import random
import torch
import torch.nn.functional as F

from drl.ddpg_agent import DDPGAgent


class DRLAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, bid_action_size, board_action_size, seed=None, checkpoint_path=None, initial_checkpoint_path=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            bid_action_size (int): Dimension of each bid action
            board_action_size (int): Dimension of each board action (number of cells in the game table)
            seed (int): Random seed of PRNG engines
            checkpoint_path (string): Directory with saved model checkpoints
        """
        self.agent = DDPGAgent(state_size, bid_action_size,
                               board_action_size, seed, checkpoint_path, initial_checkpoint_path)

    def act(self, state, train_mode=False):
        return self.agent.act(state, train_mode)

    def step(self, state, action, reward, next_state, done):
        return self.agent.step(state, action, reward, next_state, done)

    def load_model(self, checkpoint_path=None):
        """Load model's checkpoint"""
        return self.agent.load_model(checkpoint_path)

    def save_model(self, checkpoint_path=None):
        """Save model's checkpoint"""
        return self.agent.save_model(checkpoint_path)

    def get_losses(self):
        """Return list of losses obtained during training steps"""
        return self.agent.get_losses()

    def print_lr(self):
        self.agent.print_lr()
