# import numpy as np
# import torch
# import torch.nn as nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


# class FeatureBody(nn.Module):
#     def __init__(self, input_size, fc1_units=64, fc2_units=64):
#         super(FeatureBody, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Linear(input_size, fc1_units),
#             nn.LeakyReLU()
#         )

#         self.layer2 = nn.Sequential(
#             nn.Linear(fc1_units, fc2_units),
#             nn.LeakyReLU()
#         )

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         return x


# class ActorHead(nn.Module):
#     def __init__(self, input_size, action_size):
#         super(ActorHead, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Linear(input_size, action_size)
#         )

#     def forward(self, x):
#         action = self.layer1(x)
#         return action


# class CriticHead(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(ActorHead, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Linear(input_size, output_size)
#         )

#     def forward(self, x_state, action):
#         x = torch.cat((x_state, action), dim=1)
#         x = self.layer1(x)
#         return x


# class QNetworkHead(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(QNetworkHead, self).__init__()

#         self.layer1 = nn.Sequential(
#             nn.Linear(input_size, output_size),
#         )

#     def forward(self, x):
#         x = self.layer1(x)
#         return x


# class QNetwork(nn.Module):
#     def __init__(self, feature_size, action_size, feature_body):
#         super(QNetwork, self).__init__()
#         self.feature_body = feature_body
#         self.q_network_head = QNetworkHead(feature_size, action_size)

#     def forward(self, x):
#         x = self.feature_body(x)
#         x = self.q_network_head(x)
#         return x

# class Actor(nn.Module):
#     def __init__(self, feature_size, action_size, feature_body):
#         super(Actor, self).__init__()
#         self.feature_body = feature_body
#         self.actor_head = ActorHead(feature_size, action_size)

#     def forward(self, x):
#         x = self.feature_body(x)
#         x = self.actor_head(x)
#         return x

# class Critic(nn.Module):
#     def __init__(self, feature_size, action_size, feature_body):
#         super(Critic, self).__init__()
#         self.feature_body = feature_body
#         self.critic_head = CriticHead(feature_size+action_size, action_size)

#     def forward(self, x_state, action):
#         x_state = self.feature_body(x_state)
#         x = self.critic_head(x_state, action)
#         return x
