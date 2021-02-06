import argparse
from controllers.game_controller import GameController
from drl.agent import DRLAgent
from models.players.drl_player import DRLPlayer

if __name__ == '__main__':
    controller = GameController()
    drl_agent = DRLAgent(
        state_size=controller.game.N*controller.game.N+2,
        bid_action_size=1,
        board_action_size=controller.game.N*controller.game.N,
        seed=None,
        checkpoint_path=None
    )
    drl_player = DRLPlayer(0, controller.game, drl_agent, train_mode=True)
    controller.init_train(drl_player)