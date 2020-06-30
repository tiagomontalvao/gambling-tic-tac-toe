from colorama import Fore, Back, Style

class ConsoleGameView:
    BOARD_PADDING = 13
    BOARD_COLOR = Fore.GREEN
    CELL_CHAR = ' ', 'X', 'O'

    def __init__(self, board, coins_p1, coins_p2, player=None, keys=None):
        self.board = board
        self.coins_p1 = coins_p1
        self.coins_p2 = coins_p2
        self.player = player
        self.keys = keys

    def _get_cell_value(self, i, j):
        value = self.CELL_CHAR[self.board[i][j]]
        if value == ' ' and self.keys is not None:
            value = Style.RESET_ALL + Style.DIM + self.keys[i][j] + Style.RESET_ALL + self.BOARD_COLOR + Style.BRIGHT
        return value

    def _update_values(self, **kw):
        if self.player == 1: self.player = 2
        if self.player == 2: self.player = 1
        for key, value in kw.items():
            setattr(self, key, value)

    def _board_view(self):
        ret, N = '', len(self.board)
        for i in range(N):
            ret += ' '*self.BOARD_PADDING + '|'.join([' ' + self._get_cell_value(i, j) + ' ' for j in range(N)]) + '\n'
            if i < N-1:
                ret += ' '*self.BOARD_PADDING + '+'.join(['---' for _ in range(N)]) + '\n'
        return self.BOARD_COLOR + Style.BRIGHT + ret + Style.RESET_ALL

    def _players_view(self):
        ret = ''
        for i in range(1, 3):
            coins = getattr(self, f'coins_p{i}')
            ret += 'Player {}: {:3}'.format(i, coins)
            ret += [' '*11, '\n'][i-1]
        if self.player is not None:
            ret += ' '*ret.find('Player {}'.format(self.player)) + '-'*self.BOARD_PADDING + '\n'
        return ret

    def render_view(self):
        """Prints view of current model to the console"""
        view = ''
        # board
        view += self._board_view() + '\n'
        # players
        view += self._players_view() + '\n'
        # render view
        print(view)

    def update_view(self, **kw):
        """Update view properties and re-render this view"""
        self._update_values(**kw)
        self.render_view()

if __name__ == '__main__':
    board = [[0,1,0],[0,1,0],[0,0,2]]
    keys = [['q','w','e'],['a','s','d'],['z','x','c']]
    coins_p1, coins_p2, player = 105, 95, None
    console_game_view = ConsoleGameView(board, coins_p1, coins_p2, player, keys)
    console_game_view.render_view()