import platform

def get_os():
    return platform.system()

def is_windows():
    return get_os().lower() == 'windows'

def print_not_train(*args, train_mode, **kwargs):
    if not train_mode:
        print(*args, **kwargs)
