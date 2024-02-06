import sys

class CustomPrint:
    COLORS = {
        "HEADER": '\033[95m',
        "OKBLUE": '\033[94m',
        "OKGREEN": '\033[92m',
        "WARNING": '\033[93m',
        "FAIL": '\033[91m',
        "ENDC": '\033[0m',
        "BOLD": '\033[1m',
        "UNDERLINE": '\033[4m',
    }

    def __init__(self, verbosity=1):
        self.verbosity = verbosity

    def print(self, message, level=1, color=None):
        if level <= self.verbosity:
            if color and color in self.COLORS:
                formatted_message = f"{self.COLORS[color]}{message}{self.COLORS['ENDC']}"
                print(formatted_message)
            else:
                print(message)