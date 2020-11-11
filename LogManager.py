from os import system

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def log_print_v2(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def color_list():
    return ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']


def displayLog(InValue, InColor='green', attrs='bold'):
    log_print(InValue, color=InColor, attrs=[attrs])
