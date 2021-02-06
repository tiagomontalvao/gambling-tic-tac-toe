import numpy as np
import torch
from typing import List

from drl.agent import DRLAgent
from models.move import Move
from models.players.base_player import BasePlayer

class DRLPlayer(BasePlayer):
    def __init__(self, player, game, agent, train_mode=False):
        super(DRLPlayer, self).__init__(player)

        # checkpoint_path='../../drl/checkpoints/checkpoint.pt'
        checkpoint_path = None
        # seed = player if train_mode else None
        seed = None

        self.game_N = game.N
        self.sum_coins = sum(game.coins)

        self.player = player
        self.agent = agent
        # self.agent = DRLAgent(state_size=game.N*game.N+2, bid_action_size=1, board_action_size=game.N*game.N, seed=seed, checkpoint_path=checkpoint_path)
        self.train_mode = train_mode

        self.curr_state = None
        self.curr_action = None
        self.last_state = None
        self.last_action = None

        self.bid = None
        self.board_move = None

    def get_bid(self, game):
        """Update agent's state, make action, save new state and return bid"""
        self.curr_state = self._format_state_to_agent(game.get_state_array())

        self._get_experience_tuple_then_step(reward=0, done=False)

        # get action
        self.curr_action = self.agent.act(self.curr_state)

        self.last_state = self.curr_state
        self.last_action = self.curr_action

        self.bid, self.board_move = self._get_action_from_agent(self.curr_action, game.coins[self.player])

        return self.bid

    def get_board_move(self, game):
        """
        Return move already saved in self.board_move.
        It is saved with a number from 0 to game.N**2-1 (8, if N=3) and is converted to Move(x, y).
        """
        x = self.board_move // self.game_N
        y = self.board_move % self.game_N

        return Move(x, y)

    def sinalize_done(self, winner):
        """Perform step in the agent if self.train_mode=True after game is finished"""
        reward = 1 if winner == self.player else -1
        self._get_experience_tuple_then_step(reward=reward, done=True)

    def _get_experience_tuple_then_step(self, reward, done):
        """Perform step in the agent if self.train_mode=True"""
        if self.train_mode and self.last_state is not None:
            # add tuple to agent
            last_state = self._format_state_to_agent(self.last_state)
            last_action = self._format_action_to_agent(self.bid, self.board_move)
            curr_state = self._format_state_to_agent(self.curr_state)
            self.agent.step(last_state, last_action, reward, curr_state, done)

    def _format_state_to_agent(self, state: list) -> torch.Tensor:
        state = torch.Tensor(state)
        state[:2] /= self.sum_coins
        return state

    def _format_action_to_agent(self, bid: int, board_move: int) -> torch.Tensor:
        # Normalize bid
        bid /= self.sum_coins
        # Create one-hot from board_move
        board_move_array = [0] * (self.game_N ** 2)
        board_move_array[board_move] = 1
        action = torch.Tensor([bid, *board_move_array])
        return action

    # def _get_state_from_agent(self, state: torch.Tensor) -> torch.Tensor:
    #     state[:2] *= self.sum_coins
    #     return state

    def _get_action_from_agent(self, action: torch.Tensor, player_coins: int) -> List[int]:
        action = action.squeeze(0)
        # Denormalize bid
        bid = int(action[0].item() * self.sum_coins)
        bid = np.clip(bid, 0, player_coins)
        # Get board_move from softmax output
        board_move = action[1:].argmax().item()
        return bid, board_move

