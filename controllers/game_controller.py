import copy
import glob
import os

from models.game import Game
from models.move import Move
from views.console_game_view import ConsoleGameView

INITIAL_COINS_P1 = 100
INITIAL_COINS_P2 = 100
MAX_EQUAL_BIDS = 3

class GameController:
    def __init__(self):
        self.game = Game(None)
        self.view = ConsoleGameView(copy.copy(self.game.board), [INITIAL_COINS_P1, INITIAL_COINS_P2])

    def init_game(self):
        self.players = [None] + [self._select_player(Game.PLAYERS[player]) for player in range(1, 3)]

        keys = Move.KEYS \
            if self.players[1].__class__.__name__ == 'HumanPlayer' or self.players[2].__class__.__name__ == 'HumanPlayer' \
            else None

        os.system('clear')
        self.view.update_view(keys=keys)

        equal_bids = 0
        finished, winner = self.game.game_finished()
        while not finished:
            if equal_bids == MAX_EQUAL_BIDS:
                break

            bids = [None]
            for player in range(1, 3):
                while True:
                    # print(f'Player {player} bidding: ', end='')
                    print(f'Player {self.players[player].__class__.__name__}({Game.PLAYERS[player]}) bidding: ', end='')
                    bid = self.players[player].get_bid()
                    print(bid)
                    if self.game.validate_bid(bid, player):
                        bids.append(bid)
                        break
                    print(f'Invalid bid value. It should be in the range [0, {self.game.coins[player]}]...')

            if bids[1] == bids[2]:
                equal_bids += 1
                if equal_bids < MAX_EQUAL_BIDS:
                    print(f'Equal bids. Retry #{equal_bids}/{MAX_EQUAL_BIDS-1}')
                continue
            else:
                equal_bids = 0
                if bids[1] > bids[2]: player = 1
                else: player = 2

            print(f'Player {self.players[player].__class__.__name__}({Game.PLAYERS[player]}) won the bet')
            self.game.play(self.players[player].get_board_move(self.game.valid_moves()), player, bids)

            os.system('clear')
            self.view.update_view(coins=self.game.coins)

            finished, winner = self.game.game_finished()

        self._end_game(winner)


    def _end_game(self, winner):
        print('')
        if winner == Game.PLAYERS[1]:
            print('Player ' + self.players[1].__class__.__name__ + '(' + Game.PLAYERS[1] + ') won')
        elif winner == Game.PLAYERS[2]:
            print('Player ' + self.players[2].__class__.__name__ + '(' + Game.PLAYERS[2] + ') won')
        else:
            print('Game ended in a tie')

    def _select_player(self, player):
        players = glob.glob('./models/players/*_player.py')
        print('Select one of the following players to be player', player)

        for idx, player in enumerate(players):
            print(f'{idx} - {player}')

        player = 0# input("Input desired player number: ")
        module_globals = {}
        exec(open(players[int(player)]).read(), module_globals)
        return module_globals[list(module_globals.keys())[len(module_globals.keys()) - 1]](player)