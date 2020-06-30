class Move:
    KEYS = [['q','w','e'],['a','s','d'],['z','x','c']]
    def __init__(self, x, y=None, kind='coord'):
        """
        Creates a Move object, responsible for placing a player's move into the board.
        It can receive one of:
            . the coordinates of the movement in the board: {0,1,2}x{0,1,2}
            . a letter representing the movement in the board, together with kind='letter' or kind='human'
        """
        if kind=='letter' or kind=='human':
            if y is not None: raise Exception(f'In kind {kind}, it should receive only a letter representing the movement.')
            try:
                x, y = [(index, row.index(x)) for index, row in enumerate(self.KEYS) if x in row][0]
            except IndexError:
                msg = f'Letter {x} not accepted'
                print(msg)
                raise ValueError(msg)
        self.x = x
        self.y = y

    def __getitem__(self, i):
        if i == 0: return x
        if i == 1: return y

    def __eq__(self, other):
        return (self and other and self.x == other.x and self.y == other.y)
