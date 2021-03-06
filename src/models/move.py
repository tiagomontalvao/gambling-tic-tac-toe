class Move:
    """Move class, responsible for placing a player's move into the board."""

    KEYS = [[str(i*3+j+1) for j in range(3)] for i in range(3)]

    def __init__(self, x, y=None, kind='coord'):
        """
        Creates a Move object, responsible for placing a player's move into the board.
        It can receive one of:
            . the coordinates of the movement in the board: {0,1,2}x{0,1,2}
            . a letter representing the movement in the board, together with kind='human'
        """
        if kind == 'human':
            if y is not None:
                raise Exception(
                    f'In kind {kind}, it should receive only a letter representing the movement.')
            try:
                x, y = [(index, row.index(x))
                        for index, row in enumerate(self.KEYS) if x in row][0]
            except IndexError:
                msg = f'Position {x} not accepted'
                print(msg)
                raise ValueError(msg)
        elif y is None:
            raise ValueError(
                'Both row and column should be provided if agent is not human')
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return [self.x, self.y][i]

    def __eq__(self, other):
        return (self is not None and other is not None and self.x == other.x and self.y == other.y)

    def __str__(self):
        return f'Move({self.x}, {self.y})'
