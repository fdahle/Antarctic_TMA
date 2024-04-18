# Package imports
import re
from typing import Optional

# Constants
MIN_FOCAL_LENGTH = 140  # in mm
MAX_FOCAL_LENGTH = 160  # in mm


def find_focal_length(text: str) -> Optional[float]:
    """
    Extracts a focal length from the text. Only returns a result if ONE match is found, otherwise None is
    returned. As the text extraction can be incorrect, also values that fall out of the usual range of
    1xx.xx mm will be considered and tweaked to fit in this range.
    Args:
        text (str): A string that may contain focal lengths, separated by semicolons.
    Returns:
        Optional[float]: The focal length if exactly one valid focal length is found
            within the range; otherwise, None.
    Example:
        >>> find_focal_length("Camera specs;50.25; Zoom level.")
        150.25
        >>> find_focal_length("Lens details: 50.256;another part.")
        150.256
        >>> find_focal_length("Fl:50.8; Focal length: 50.9.")
        None
    """

    # split the text into their respective boxes
    text_per_box = text.split(";")

    # init already a list to keep possible focal lengths
    all_matches = []

    # Create a pattern for regex to extract the focal length
    pattern = r'5\d\.\d{2,3}\d?'

    # Iterate all text segments
    for text_part in text_per_box:

        # try to find a match with regex
        matches = re.findall(pattern, text_part)
        all_matches.extend(matches)

    # check if the correct number of matches is found
    if len(all_matches) == 1:

        # convert the focal length to a float
        focal_length = float(all_matches[0])

        # increase focal length to 1xx.xx
        if focal_length < 100:
            focal_length = 100 + focal_length

        # check if focal length is correct
        if focal_length < MIN_FOCAL_LENGTH or focal_length > MAX_FOCAL_LENGTH:
            return None

        return focal_length

    return None
