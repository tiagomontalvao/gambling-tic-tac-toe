import platform

def get_os():
    return platform.system()

def is_windows():
    return get_os().lower() == 'windows'
