import colorama
from colorama import Fore

colorama.init(autoreset=True)


def print_v(txt, verbose=True, color=None, pbar=None):
    """
    print_v(txt, verbose, color, pbar)
    This small function is a replacement function for the classical print statement. It checks
    automatically if a text should be printed or applies colour to the text. Furthermore, it can
    append text to a tqdm progress-bar
    Args:
        txt (String): The text that should be printed
        verbose (Boolean, True): If false, nothing will be printed
        color (String, None): The color of the text. If it is none, no colour will be applied
        pbar (tqdm-progressbar, None): If this is defined, the text will not be printed
            regularly, but attached to a progressbar
    Returns:
         -
    """

    # easiest case -> we don't want to print
    if verbose is False:
        return

    # convert text to str
    txt = str(txt)

    if color == "red":
        txt = Fore.RED + txt
    elif color == "green":
        txt = Fore.GREEN + txt
    elif color == "yellow":
        txt = Fore.YELLOW + txt
    elif color == "black":
        txt = Fore.BLACK + txt

    if pbar is None:
        print(txt)
    else:
        pbar.set_postfix_str(txt)

    return
