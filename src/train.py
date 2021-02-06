import argparse
from controllers.game_controller import GameController

if __name__ == '__main__':
    controller = GameController()
    controller.init_train()