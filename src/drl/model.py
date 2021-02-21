import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

# class Network(nn.Module):
#     """
#     Network Model used by the agent.
#     Input:
#         - state
#     Output:
#         - bid to make
#         - board move to make
#         - V-value estimate
#     """

#     def __init__(self, state_size, board_size, fc1_units=32, seed=0):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Size of state (board, coins)
#             board_size (int): Size of game board
#             seed (int): Random seed
#         """
#         super(Network, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.state_size = state_size
#         self.board_size = board_size

#         self.layer1 = nn.Sequential(
#             nn.Linear(state_size, fc1_units),
#             nn.BatchNorm1d(num_features=fc1_units),
#             nn.LeakyReLU()
#         )

#         self.layer_output_bid = nn.Linear(fc1_units, 1)
#         self.layer_output_board_move = nn.Linear(fc1_units, board_size)
#         self.layer_output_value = nn.Linear(fc1_units, 1)

#     def forward(self, state):
#         """Evaluate the NN in the given state. Returns the tuple (bid, board_move, V-value estimate)"""

#         # Hidden Layers
#         x = state.unsqueeze(0)
#         x = self.layer1(state)

#         # Output layer
#         bid = self.layer_output_bid(x)
#         move_probs = self.layer_output_board_move(x)
#         value = self.layer_output_value(x)

#         avail = (state[:,:self.board_size] == -1).type(torch.FloatTensor).to(device)

#         # Set probability of invalid board moves to 0
#         max_logit = torch.max(move_probs)

#         # TODO: analyze need for this \/
#         # subtract off max for numerical stability (avoids blowing up at infinity)
#         exp = torch.exp(move_probs-max_logit)
#         exp = avail*exp
#         prob = exp/torch.sum(exp)
#         print(prob)

#         return bid, prob, value


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
            # nn.BatchNorm1d(num_features=fc1_units),
            nn.PReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc1_units, fc2_units),
            # nn.BatchNorm1d(num_features=fc2_units),
            nn.PReLU()
        )
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
        self.fc3_bid[0].weight.data.uniform_(-3e-3, 3e-3)
        self.fc3_board.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        x = self.fc2(x)
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
            # nn.BatchNorm1d(num_features=fcs1_units),
            nn.PReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fcs1_units+action_size, fc2_units),
            # nn.BatchNorm1d(num_features=fc2_units),
            nn.PReLU()
        )
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1[0].weight.data.uniform_(*hidden_init(self.fcs1[0]))
        self.fc2[0].weight.data.uniform_(*hidden_init(self.fc2[0]))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fcs1(state)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        return self.fc3(x)

# class Actor(nn.Module):
#     def __init__(self, state_size=11, board_action_size=9, bid_action_size=1, fc1_units=64, fc2_units=64):
#         super(Actor, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Linear(state_size, fc1_units),
#             nn.PReLU()
#         )

#         self.layer2 = nn.Sequential(
#             nn.Linear(fc1_units, fc2_units),
#             nn.PReLU()
#         )

#         self.layer3 = nn.Sequential(
#             nn.Linear(fc2_units, action_size)
#         )

#     def forward(self, state):
#         x = self.layer1(state)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x


# class Critic(nn.Module):
#     def __init__(self, state_size=11, action_size=10, fc1_units=64, fc2_units=64):
#         super(Actor, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Linear(state_size, fc1_units),
#             nn.PReLU()
#         )

#         self.layer2 = nn.Sequential(
#             nn.Linear(fc1_units+action_size, fc2_units),
#             nn.PReLU()
#         )

#         self.layer3 = nn.Sequential(
#             nn.Linear(fc2_units, output_size)
#         )

#     def forward(self, state, action):
#         xs = self.layer1(state)
#         x = torch.cat((xs, action), dim=1)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         return x


# class QNetwork(nn.Module):
#     def __init__(self, input_size, output_size, fc1_units=64, fc2_units=64):
#         super(QNetwork, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Linear(state_size, fc1_units),
#             # nn.BatchNorm1d(fc1_units),
#             nn.PReLU()
#         )

#         self.layer2 = nn.Sequential(
#             nn.Linear(fc1_units, fc2_units),
#             # nn.BatchNorm1d(fc2_units),
#             nn.PReLU()
#         )

#         self.layer3 = nn.Linear(fc2_units, action_size)

#     def forward(self, x):
#         x = self.layer1(x)
#         return x
