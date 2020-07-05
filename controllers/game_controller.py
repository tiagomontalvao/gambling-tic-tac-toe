import copy
import glob
import os

from models.game import Game
from models.move import Move
from views.console_game_view import ConsoleGameView

INITIAL_COINS = [100, 100]
MAX_EQUAL_BIDS = 3

class GameController:
    def __init__(self):
        self.game = Game()
        self.view = ConsoleGameView(self.game.get_clone(), INITIAL_COINS)

    def init_game(self):
        self.players = [self._select_player(player) for player in range(2)]

        keys = Move.KEYS \
            if self.players[0].__class__.__name__ == 'HumanPlayer' or self.players[1].__class__.__name__ == 'HumanPlayer' \
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
                print('Invalid board move. Try again')
                move = self.players[player].get_board_move(game_clone)

            # make board move
            self.game.play(move, player)

            # os.system('clear')
            game_clone = self.game.get_clone()
            self.view.update_view(game=game_clone)

            finished, winner = self.game.game_finished()

        for player in self.players:
            player.sinalize_done(winner)

        self._end_game(winner)

    def _end_game(self, winner):
        print('')
        if winner is None:
            print('Game ended in a tie')
        else:
            print('Player ' + self.players[winner].__class__.__name__ + '(' + ConsoleGameView.PLAYERS[winner] + ') won')

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
