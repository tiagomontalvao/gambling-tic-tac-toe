import numpy as np
import torch
from typing import List

from drl.agent import DRLAgent
from models.move import Move
from models.players.base_player import BasePlayer
from utils import get_reward_from_winner

REWARD_PER_STEP = -0.05


class DRLPlayer(BasePlayer):
    def __init__(self, player, game, agent=None, train_mode=False, initial_checkpoint_path='../checkpoints/model_v3.pt'):
        super().__init__(player, game)

        # seed = player if train_mode else None
        seed = None

        self.game_N = game.N
        self.sum_coins = sum(game.coins)

        self.player = player

        if agent is None:
            agent = DRLAgent(state_size=game.N*game.N+2, bid_action_size=1,
                             board_action_size=game.N*game.N, seed=seed,
                             initial_checkpoint_path=initial_checkpoint_path)
        self.agent = agent

        self.train_mode = train_mode

        self.curr_coins = game.coins[self.player]
        self.curr_state = None
        self.last_state = None
        self.last_action = None

        self.bid = None
        self.board_move = None

    def get_bid(self, game):
        """Update agent's state, make action, save new state and return bid"""
        self.curr_state = self._format_state_to_agent(game.get_state_array())

        self._get_experience_tuple_then_step(
            reward=REWARD_PER_STEP, done=False)

        # get action
        curr_action = self.agent.act(self.curr_state, self.train_mode)

        self.curr_coins = game.coins[self.player]
        self.last_state = self.curr_state
        self.last_action = curr_action

        self.bid, self.board_move = self._get_action_from_agent(curr_action)

        return self.bid

    def get_board_move(self, game):
        """
        Return move already saved in self.board_move
        It is saved with a number from 0 to game.N**2-1 (8, if N=3) and is converted to Move(x, y)
        """
        x = self.board_move // self.game_N
        y = self.board_move % self.game_N

        return Move(x, y)

    def update_epsilon(self, epsilon):
        """Update epsilon used in epsilon-greedy algorithm"""
        self.epsilon = epsilon

    def sinalize_done(self, winner):
        """Perform step in the agent if self.train_mode=True after game is finished"""
        reward = get_reward_from_winner(self.player, winner)
        self._get_experience_tuple_then_step(reward=reward, done=True)

    def _get_experience_tuple_then_step(self, reward, done):
        """Perform step in the agent if self.train_mode=True"""
        if self.train_mode and self.last_state is not None:
            # add tuple to agent
            last_state = self.last_state
            # last_action = self._format_action_to_agent(
            #     self.bid, self.board_move) # wrong approach
            last_action = self.last_action
            curr_state = self.curr_state
            self.agent.step(last_state, last_action, reward, curr_state, done)

    def _format_state_to_agent(self, state: list) -> torch.Tensor:
        state = torch.Tensor(state)
        # if player is 1, swap 0 with 1 so that is thinks it's player 0
        if self.player == 1:
            state[:2] = torch.flip(state[:2], (0,))
            state[2:][state[2:] >= 0] = 1-state[2:][state[2:] >= 0]
        state[:2] /= self.sum_coins
        return state

    # def _format_action_to_agent(self, bid: int, board_move: int) -> torch.Tensor:
    #     # Normalize bid
    #     bid /= self.curr_coins
    #     # Create one-hot from board_move # Whyyyyy?
    #     board_move_array = [0] * (self.game_N ** 2)
    #     board_move_array[board_move] = 1
    #     action = torch.Tensor([bid, *board_move_array])
    #     return action

    # def _get_state_from_agent(self, state: torch.Tensor) -> torch.Tensor:
    #     state[:2] *= self.sum_coins
    #     return state

    def _get_action_from_agent(self, action: torch.Tensor) -> List[int]:
        action = action.squeeze(0)
        # Denormalize bid

        # Relative to total coins
        bid = int(round(action[0].item() * self.sum_coins))
        bid = np.clip(bid, 0, self.curr_coins)

        # Relative to current coins
        # bid = int(round(action[0].item() * self.curr_coins))
        # bid = np.clip(bid, 0, self.curr_coins)

        # Epsilon-greedy choice of action when self.train_mode=True
        action_probs = action[1:]
        if not self.train_mode or np.random.random() > self.epsilon:
            board_move = np.argmax(action_probs)
        else:
            # Consider only available positions
            p = np.ones(len(action_probs))
            p[action_probs <= 1e-8] = 0
            p /= p.sum()
            board_move = np.random.choice(np.arange(len(action_probs)), p=p)

        # # Get board_move from softmax output
        # board_move = action[1:].argmax().item()

        return bid, board_move

    def print_lr(self):
        self.agent.print_lr()
