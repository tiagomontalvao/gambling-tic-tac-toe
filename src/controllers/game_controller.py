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
from utils import get_reward_from_winner, print_not_train, is_windows

INITIAL_COINS = [100, 100]
MAX_EQUAL_BIDS = 3

class GameController:
    def __init__(self):
        self.game = Game()
        self.view = ConsoleGameView(self.game.get_clone())

    def init_game(self):
        self.players = [self._select_player(player) for player in range(2)]

        keys = Move.KEYS \
            if any(self.players[i].__class__.__name__ == 'HumanPlayer' for i in range(2)) and is_windows() \
            else None

        # os.system('clear')
        self.view.update_view(keys=keys)

        equal_bids = 0
        finished, winner = self.game.game_finished()
        game_clone = self.game.get_clone()
        while not finished:
            if equal_bids == MAX_EQUAL_BIDS:
                break

            bids = []
            for player in range(2):
                while True:
                    print(f'Player {self.players[player].__class__.__name__}({ConsoleGameView.PLAYERS[player]}) bidding: ', end='')
                    bid = self.players[player].get_bid(game_clone)
                    print(bid)
                    if self.game.validate_bid(bid, player):
                        bids.append(bid)
                        break
                    print(f'Invalid bid value of {bid}. It should be in the range [0, {game_clone.coins[player]}]...')

            if bids[0] == bids[1] and self.game.coins[0] == self.game.coins[1]:
                equal_bids += 1
                if equal_bids < MAX_EQUAL_BIDS:
                    print(f'Equal bids with equal coins. Retry #{equal_bids}/{MAX_EQUAL_BIDS-1}')
                continue
            else:
                equal_bids = 0
                if bids[0] > bids[1] or (bids[0] == bids[1] and self.game.coins[0] > self.game.coins[1]): player = 0
                else: player = 1

            print(f'Player {self.players[player].__class__.__name__}({ConsoleGameView.PLAYERS[player]}) won the bet')

            # update players coins
            self.game.update_coins(player, bids)

            # get board move
            game_clone = self.game.get_clone()
            move = self.players[player].get_board_move(game_clone)

            # validate move
            valid_moves = game_clone.valid_moves()
            while move not in valid_moves:
                print(f'Invalid board move: ({move.x}, {move.y}). Try again')
                move = self.players[player].get_board_move(game_clone)

            # make board move
            self.game.play(move, player)

            # os.system('clear')
            game_clone = self.game.get_clone()
            self.view.update_view(game=game_clone)

            finished, winner = self.game.game_finished()

        for player in self.players:
            player.sinalize_done(winner)

        self._end_game(winner, train_mode=False)


    def init_train(self, drl_player, train_args=None):
        n_episodes = 5000
        checkpoint_each = 100
        print_each = 10

        self.view.update_view(print_function=lambda *args, **kwargs: print_not_train(*args, train_mode=True, **kwargs))
        self._load_train(drl_player)

        scores = []
        scores_window = deque(maxlen=100)
        try:
            for i in range(1, n_episodes+1):
                self.game.reset()
                score = self._run_episode(train_mode=True)
                scores.append(score)
                scores_window.append(score)

                if i%print_each == 0:
                    avg_score = np.mean(scores_window)
                    print("\rEpisode #{:3d}  |  "
                        "Score: {:+6.2f}  |  "
                        "Avg. Score: {:+6.2f}".format(i, score, avg_score)
                    )
                # if i%checkpoint_each == 0:
                #     drl_player.agent.save_model()
                #     # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pt')
                #     # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pt')
                #     print('Checkpoint saved!')
        except KeyboardInterrupt:
            pass

        # save model after all episodes
        # drl_player.agent.save_model()
        print('Model saved!')
        fig, ax = plt.subplots()
        ax.plot(scores)
        pd.Series(scores).rolling(100, 1).mean().plot(ax=ax, label='MA(100)')
        plt.show()

        return scores

    def _load_train(self, drl_player):
        self.players = [
            drl_player,
            RandomPlayer(1, self.game.get_clone())
        ]

    def _run_episode(self, train_mode=False):
        equal_bids = 0
        finished, winner = self.game.game_finished()
        game_clone = self.game.get_clone()
        while not finished:
            if equal_bids == MAX_EQUAL_BIDS:
                break

            bids = []
            for player in range(2):
                while True:
                    print_not_train(f'Player {self.players[player].__class__.__name__}({ConsoleGameView.PLAYERS[player]}) bidding: ', end='', train_mode=train_mode)
                    bid = self.players[player].get_bid(game_clone)
                    print_not_train(bid, train_mode=train_mode)
                    if self.game.validate_bid(bid, player):
                        bids.append(bid)
                        break
                    print_not_train(f'Invalid bid value of {bid}. It should be in the range [0, {game_clone.coins[player]}]...', train_mode=train_mode)

            if bids[0] == bids[1] and self.game.coins[0] == self.game.coins[1]:
                equal_bids += 1
                if equal_bids < MAX_EQUAL_BIDS:
                    print_not_train(f'Equal bids with equal coins. Retry #{equal_bids}/{MAX_EQUAL_BIDS-1}', train_mode=train_mode)
                continue
            else:
                equal_bids = 0
                if bids[0] > bids[1] or (bids[0] == bids[1] and self.game.coins[0] > self.game.coins[1]): player = 0
                else: player = 1

            print_not_train(f'Player {self.players[player].__class__.__name__}({ConsoleGameView.PLAYERS[player]}) won the bet', train_mode=train_mode)

            # update players coins
            self.game.update_coins(player, bids)

            # get board move
            game_clone = self.game.get_clone()
            move = self.players[player].get_board_move(game_clone)

            # validate move
            valid_moves = game_clone.valid_moves()
            while move not in valid_moves:
                print_not_train(f'Invalid board move: ({move.x}, {move.y}). Try again', train_mode=train_mode)
                move = self.players[player].get_board_move(game_clone)

            # make board move
            self.game.play(move, player)

            # os.system('clear')
            game_clone = self.game.get_clone()
            self.view.update_view(game=game_clone)

            finished, winner = self.game.game_finished()

            # if train_mode:
            #     state = None
            #     action = None
            #     reward = get_reward_from_winner(0, winner)
            #     next_state = self.game.get_state_array()
            #     done = finished
            #     self.players[0].step(self, state, action, reward, next_state, done)

        for player in self.players:
            player.sinalize_done(winner)

        self._end_game(winner, train_mode)

        return get_reward_from_winner(0, winner)


    def _end_game(self, winner, train_mode):
        print_not_train('', train_mode=train_mode)
        if winner is None:
            print_not_train('Game ended in a tie', train_mode=train_mode)
        else:
            print_not_train('Player ' + self.players[winner].__class__.__name__ + '(' + ConsoleGameView.PLAYERS[winner] + ') won', train_mode=train_mode)

    def _select_player(self, player):
        players = glob.glob('./models/players/*_player.py')
        # filter abstract base player
        players = [player_name for player_name in players if 'base_player' not in player_name]

        print('Select one of the following players to be player', player+1)

        for idx, player_file in enumerate(players):
            print(f'{idx} - {player_file}')

        chosen_player = input("Input desired player number: ")
        module_globals = {}
        exec(open(players[int(chosen_player)]).read(), module_globals)
        return module_globals[list(module_globals.keys())[len(module_globals.keys()) - 1]](player, self.game.get_clone())
