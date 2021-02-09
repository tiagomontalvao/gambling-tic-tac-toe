import platform

def get_os():
    return platform.system()

def is_windows():
    return get_os().lower() == 'windows'

def print_not_train(*args, train_mode, **kwargs):
    if not train_mode:
        print(*args, **kwargs)

def get_reward_from_winner(player, winner):
    if winner is None: return 0
    if player == winner: return 1
    if player == 1-winner: return -1
    raise RuntimeError("winner should be None, 0 or 1")