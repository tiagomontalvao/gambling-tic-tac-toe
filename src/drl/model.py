'''
MIT License

Copyright (c) 2018 Udacity

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
This code is taken from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/model.py,
with slight modifications to it.
'''

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=11, bid_action_size=1, board_action_size=9, fc1_units=400, fc2_units=300, seed=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            bid_action_size (int): Dimension of each bid action
            board_action_size (int): Dimension of each board action (number of cells in the game table)
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            seed (int): Random seed of PRNG engines
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.board_action_size = board_action_size
        self.fc1 = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.PReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            nn.PReLU()
        )
        # self.fc_aux = nn.Sequential(
        #     nn.Linear(fc2_units, fc2_units),
        #     nn.PReLU()
        # )
        self.fc3_bid = nn.Sequential(
            nn.Linear(fc2_units, bid_action_size),
            nn.Sigmoid()
        )
        # self.fc3_board = nn.Sequential(
        #     nn.Linear(fc2_units, board_action_size),
        #     nn.Softmax(dim=1)
        # )
        self.fc3_board = nn.Linear(fc2_units, board_action_size)
        self.softmax_board = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1[0].weight.data.uniform_(*hidden_init(self.fc1[0]))
        self.fc2[0].weight.data.uniform_(*hidden_init(self.fc2[0]))
        # self.fc_aux[0].weight.data.uniform_(*hidden_init(self.fc_aux[0]))
        self.fc3_bid[0].weight.data.uniform_(-3e-3, 3e-3)
        self.fc3_board.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = self.fc2(x)
        # x = self.fc_aux(x)  # testing one more layer

        x_bid = self.fc3_bid(x)
        x_board = self.fc3_board(x)

        avail = (state[:, 2:] == -1).type(torch.FloatTensor).to(device)
        x_board[avail == 0] = -float('inf')

        x_board = self.softmax_board(x_board)
        out = torch.cat((x_bid, x_board), dim=1)

        return out


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size=11, action_size=10, fcs1_units=400, fc2_units=300, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            seed (int): Random seed
        """
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.fcs1 = nn.Sequential(
            nn.Linear(state_size, fcs1_units),
            nn.PReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fcs1_units+action_size, fc2_units),
            nn.PReLU()
        )
        # self.fc_aux = nn.Sequential(
        #     nn.Linear(fc2_units, fc2_units),
        #     nn.PReLU()
        # )
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1[0].weight.data.uniform_(*hidden_init(self.fcs1[0]))
        self.fc2[0].weight.data.uniform_(*hidden_init(self.fc2[0]))
        # self.fc_aux[0].weight.data.uniform_(*hidden_init(self.fc_aux[0]))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fcs1(state)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        # x = self.fc_aux(x)  # testing one more layer
        return self.fc3(x)
