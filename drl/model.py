import numpy as np
import torch
import torch.nn as nn

class Network(nn.Module):
    """
    Network Model used by the agent.
    Input:
        - state
    Output:
        - bid to make
        - board move to make
        - V-value estimate
    """

    def __init__(self, state_size, board_size, fc1_units=32, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Size of state (board, coins)
            board_size (int): Size of game board
            seed (int): Random seed
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.layer1 = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.BatchNorm1d(num_features=fc1_units),
            nn.LeakyReLU()
        )

        self.layer_output_bid = nn.Linear(fc1_units, 1)
        self.layer_output_board_move = nn.Linear(fc1_units, board_size)
        self.layer_output_value = nn.Linear(fc1_units, 1)

    def forward(self, state):
        """Evaluate the NN in the given state. Returns the tuple (bid, board_move, V-value estimate)"""

        # Hidden Layers
        x = state.unsqueeze(0)
        x = self.layer1(state)

        # Output layer
        bid = self.layer_output_bid(x)
        board_move = self.layer_output_board_move(x)
        value = self.layer_output_value(x)

        return bid, board_move, value
