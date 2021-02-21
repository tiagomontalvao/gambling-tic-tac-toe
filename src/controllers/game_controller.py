import copy
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque

from models.game import Game
from models.move import Move
from models.players.drl_player import DRLPlayer
from models.players.random_player import RandomPlayer
from views.console_game_view import ConsoleGameView
from utils import get_reward_from_winner, print_not_train_or_eval, is_windows

INITIAL_COINS = [100, 100]
DEBUG = False


class GameController:
    def __init__(self):
        self.game = Game()
        self.view = ConsoleGameView(self.game.get_clone())

    def init_game(self, drl_checkpoint_path=None):
        self.players = [self._select_player(
            player, drl_checkpoint_path=drl_checkpoint_path) for player in range(2)]

        keys = Move.KEYS \
            if any(self.players[i].__class__.__name__ == 'HumanPlayer' for i in range(2)) \
            else None

        self.view.update_view(keys=keys)
        return self.run_episode(train_mode=False)

    def reset_game(self, drl_checkpoint_path=None):
        self.game.reset()
        self.view = ConsoleGameView(self.game.get_clone())
        self.init_game(drl_checkpoint_path=drl_checkpoint_path)

    def run_episode(self, train_mode=False, eval_mode=False):
        finished, winner = self.game.game_finished()
        game_clone = self.game.get_clone()
        while not finished:
            bids = []
            for player in range(2):
                while True:
                    print_not_train_or_eval(
                        f'{self._player_str(player)} bidding...',
                        train_mode=train_mode, eval_mode=eval_mode)
                    bid = self.players[player].get_bid(game_clone)
                    if self.game.validate_bid(bid, player):
                        bids.append(bid)
                        break
                    print_not_train_or_eval(
                        f'Invalid bid value of {bid}. It should be in the range [0, {game_clone.coins[player]}]...',
                        train_mode=train_mode, eval_mode=eval_mode)

            if DEBUG and (train_mode or eval_mode):
                print('Bids: {}, {}'.format(bids[0], bids[1]))
                print('Board moves: {}, {}'.format(self.players[0].get_board_move(
                    game_clone), self.players[1].get_board_move(game_clone)))

            if bids[0] == bids[1] and self.game.coins[0] == self.game.coins[1]:
                player_bid_winner = np.random.randint(2)
                print_not_train_or_eval(
                    f'Equal bids with equal coins. Randomly chosen player: {self._player_str(player_bid_winner)}', train_mode=train_mode, eval_mode=eval_mode)
            else:
                if bids[0] > bids[1] or (bids[0] == bids[1] and self.game.coins[0] > self.game.coins[1]):
                    player_bid_winner = 0
                else:
                    player_bid_winner = 1

            for player in range(2):
                print_not_train_or_eval(
                    f'{self._player_str(player)} bid: {bids[player]}',
                    end='  |  ', train_mode=train_mode, eval_mode=eval_mode)
            print_not_train_or_eval(train_mode=train_mode, eval_mode=eval_mode)

            print_not_train_or_eval(
                f'{self._player_str(player_bid_winner)} won the bet',
                train_mode=train_mode, eval_mode=eval_mode)

            # update players coins
            self.game.update_coins(player_bid_winner, bids)

            # get board move
            game_clone = self.game.get_clone()
            move = self.players[player_bid_winner].get_board_move(game_clone)

            # validate move
            valid_moves = game_clone.valid_moves()
            while move not in valid_moves:
                print_not_train_or_eval(
                    f'Invalid board move: ({move.x}, {move.y}). Try again', train_mode=train_mode, eval_mode=eval_mode)
                move = self.players[player_bid_winner].get_board_move(
                    game_clone)

            # make board move
            self.game.play(move, player_bid_winner)

            game_clone = self.game.get_clone()
            self.view.update_view(game=game_clone)

            finished, winner = self.game.game_finished()

        for player in self.players:
            player.sinalize_done(winner)

        self._end_game(winner, train_mode, eval_mode)

        return winner

    def load_train(self, players):
        self.view.update_view(print_function=lambda *args, **kwargs: 0)
        self.players = players

    def load_validation(self, players):
        self.view.update_view(print_function=lambda *args, **kwargs: 0)
        self.players = players

    def _end_game(self, winner, train_mode, eval_mode):
        print_not_train_or_eval('', train_mode=train_mode, eval_mode=eval_mode)
        if winner is None:
            print_not_train_or_eval(
                'Game ended in a tie', train_mode=train_mode, eval_mode=eval_mode)
        else:
            print_not_train_or_eval(
                f'{self._player_str(winner)} won', train_mode=train_mode, eval_mode=eval_mode)

    def _player_str(self, player):
        return f'Player {1+player} {self.players[player].__class__.__name__}({ConsoleGameView.PLAYERS[player]})'

    def _select_player(self, player, drl_checkpoint_path=None):
        players = glob.glob('./models/players/*_player.py')
        # filter abstract base player
        players = [
            player_name for player_name in players if 'base_player' not in player_name]

        print('Select one of the following players to be player', player+1)

        for idx, player_file in enumerate(players):
            print(f'{idx} - {player_file}')

        chosen_player = input("Input desired player number: ")
        module_globals = {}
        exec(open(players[int(chosen_player)]).read(), module_globals)
        player_class = module_globals[list(module_globals.keys())[
            len(module_globals.keys()) - 1]]
        player_kwargs = {}
        print(
            f'player_class.__class__.__name__ = {player_class.__class__.__name__}')
        if player_class.__class__.__name__ == 'DRLPlayer':
            print(
                f"player_kwargs['initial_checkpoint_path'] = drl_checkpoint_path = {drl_checkpoint_path}")
            player_kwargs['initial_checkpoint_path'] = drl_checkpoint_path
        return player_class(player, self.game.get_clone(), **player_kwargs)
