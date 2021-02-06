from colorama import Fore, Back, Style
from utils import is_windows

class ConsoleGameView:
    BOARD_PADDING = 17
    PLAYERS = 'X', 'O'
    CELL_CHAR = ' ', *PLAYERS
    BOARD_COLOR = '' if is_windows() else Fore.GREEN
    RESET_ALL = '' if is_windows() else Style.RESET_ALL
    BRIGHT = '' if is_windows() else Style.BRIGHT
    DIM = '' if is_windows() else Style.DIM

    def __init__(self, game, keys=None):
        self.game = game
        self.keys = keys

    def _get_cell_value(self, i, j):
        value = self.CELL_CHAR[self.game.board[i][j]+1]
        if value == ' ' and self.keys is not None:
            value = self.RESET_ALL + self.DIM + self.keys[i][j] + self.RESET_ALL + self.BOARD_COLOR + self.BRIGHT
        return value

    def _update_values(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)

    def _board_view(self):
        ret, N = '', len(self.game.board)
        for i in range(N):
            ret += ' '*self.BOARD_PADDING + '|'.join([' ' + self._get_cell_value(i, j) + ' ' for j in range(N)]) + '\n'
            if i < N-1:
                ret += ' '*self.BOARD_PADDING + '+'.join(['---' for _ in range(N)]) + '\n'
        return self.BOARD_COLOR + self.BRIGHT + ret + self.RESET_ALL

    def _players_view(self):
        ret = ''
        for player in range(2):
            coins = self.game.coins[player]
            ret += 'Player {} ({}): {:3}'.format(player+1, self.CELL_CHAR[player+1], coins)
            ret += [' '*11, '\n'][player]
        return ret

    def render_view(self):
        """Prints view of current model to the console"""
        view = ''
        # board
        view += self._board_view()
        # players
        view += self._players_view()
        # render view
        print(view)

    def update_view(self, **kw):
        """Update view properties and re-render this view"""
        self._update_values(**kw)
        self.render_view()

if __name__ == '__main__':
    from models.game import Game
    board = [[0,1,0],[0,1,0],[0,0,2]]
    coins = [105, 95]
    game = Game(board, coins)
    keys = [[1,2,3],[4,5,6],[7,8,9]]
    console_game_view = ConsoleGameView(game, keys)
    console_game_view.render_view()