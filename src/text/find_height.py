"""Extracts a height from the text"""

# Library imports
import re
from typing import Optional

# Constants
MIN_HEIGHT = 10000  # in feet
MAX_HEIGHT = 30000  # in feet


def find_height(text: str) -> Optional[int]:
    """
    Extracts a height from the text that falls within the specified range. Only returns a result if ONE match is found,
    otherwise None is returned.
    Args:
        text (str): A string that may contain various heights, separated by semicolons.
    Returns:
        Optional[int]: The height if exactly one valid height is found within the range; otherwise, None.
    Example:
        >>> find_height("The aircraft flies at 25000; cruising altitude.")
        25000
        >>> find_height("Elevation levels: 10000; 15000; 20000.")
        None
    """

    # split the text into their respective boxes
    text_per_box = text.split(";")

    # init already a list to keep possible heights
    all_matches = []

    # Create a pattern for regex to extract the height
    pattern = r"\b(" + "|".join([str(i) for i in range(MIN_HEIGHT, MAX_HEIGHT + 1, 100)]) + r")\b"

    # iterate all text segments
    for text_part in text_per_box:

        # try to find a match with regex
        matches = re.findall(pattern, text_part)
        all_matches.extend(matches)

    # check if the correct number of matches is found
    if len(all_matches) == 1:

        # return height as integer
        height = all_matches[0]
        return int(height)

    else:
        return None
