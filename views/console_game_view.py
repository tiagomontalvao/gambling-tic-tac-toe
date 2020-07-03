from colorama import Fore, Back, Style

class ConsoleGameView:
    BOARD_PADDING = 17
    BOARD_COLOR = Fore.GREEN
    CELL_CHAR = ' ', 'X', 'O'

    def __init__(self, game, keys=None):
        self.game = game
        self.keys = keys

    def _get_cell_value(self, i, j):
        value = self.game.board[i][j]
        if value == ' ' and self.keys is not None:
            value = Style.RESET_ALL + Style.DIM + self.keys[i][j] + Style.RESET_ALL + self.BOARD_COLOR + Style.BRIGHT
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
        return self.BOARD_COLOR + Style.BRIGHT + ret + Style.RESET_ALL

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
    keys = [['q','w','e'],['a','s','d'],['z','x','c']]
    console_game_view = ConsoleGameView(game, keys)
    console_game_view.render_view()