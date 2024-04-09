import re

MIN_HEIGHT = 10000  # in feet
MAX_HEIGHT = 30000  # in feet


def find_height(text):

    # split the text into their respective boxes
    text_per_box = text.split(";")

    for text_part in text_per_box:

        pattern = r"\b(" + "|".join([str(i) for i in range(MIN_HEIGHT, MAX_HEIGHT + 1, 100)]) + r")\b"

        matches = re.findall(pattern, text_part)

    if len(matches) == 1:
        return matches[0]
