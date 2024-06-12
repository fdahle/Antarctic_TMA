"""find the cam id in a text"""

# Library imports
import re
from typing import Optional


def find_cam_id(text: str) -> Optional[str]:
    """
    Extracts a cam id from the text. Only returns a result if ONE match is found, otherwise None is
    returned. A cam id must be in the patter XX-XXX where X is a digit.
    Args:
        text (str): A string that may contain cam ids, separated by semicolons.
    Returns:
        Optional[str]: The focal length if exactly one valid cam id is found
            within the range; otherwise, None.
    """
    # split the text into their respective boxes
    text_per_box = text.split(";")

    # init already a list to keep possible cam_ids
    all_matches = []

    # Create a pattern for regex to extract the cam_id
    pattern = r"5\d-[0-9]{3}"

    # Iterate all text segments
    for text_part in text_per_box:
        # try to find a match with regex
        matches = re.findall(pattern, text_part)
        all_matches.extend(matches)

    # check if the correct number of matches is found
    if len(all_matches) == 1:
        return all_matches[0]
    else:
        return None
