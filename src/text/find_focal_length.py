"""find the focal length in a text"""

# Library imports
import re
from typing import Optional

# Constants
MIN_FOCAL_LENGTH = 140  # in mm
MAX_FOCAL_LENGTH = 160  # in mm

def find_focal_length(text: str, method="numeric") -> Optional[float]:
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
    if method == "numeric":
        pattern = r'5\d\.\d{2,3}\d?'
    elif method == "text":
        pattern = r'(?=([^\s]{3}\.[^\s]{3}))'
        pattern = r'(?<!\w)(?=[A-Za-z0-9]*\d)[A-Za-z]*?(\d{1,3}\.\d{1,3})(?=\D|$)'

    # Iterate all text segments
    for text_part in text_per_box:


        candidates = re.findall(pattern, text_part)
        print(candidates)
        for val in candidates:
            if method == "numeric":
                if val is not None and val < 100:
                    val += 100
                if val is not None and MIN_FOCAL_LENGTH <= val <= MAX_FOCAL_LENGTH:
                    all_matches.append(val)
            elif method == "text":
                # Require at least 3 digit characters in the candidate
                digit_count = sum(c.isdigit() for c in val.replace('.', ''))
                if digit_count >= 4:
                    all_matches.append(val)

    # check if the correct number of matches is found
    if len(all_matches) == 1:

        # convert the focal length to a float
        if method == "numeric":
            focal_length = float(all_matches[0])

            # increase focal length to 1xx.xx
            if focal_length < 100:
                focal_length = 100 + focal_length

            # check if focal length is correct
            if focal_length < MIN_FOCAL_LENGTH or focal_length > MAX_FOCAL_LENGTH:
                return None

        elif method == "text":
            # if all matches are identical, just keep the first one
            if len(set(all_matches)) == 1:
                all_matches = [all_matches[0]]
            print(all_matches)
            focal_length = all_matches[0]

            if '.' not in focal_length:
                raise ValueError("Input must contain a decimal point.")

            left, right = focal_length.split('.')

            left_padded = left.rjust(3, 'x')
            right_padded = right.ljust(3, 'x')
            focal_length = f"{left_padded}.{right_padded}"

        else:
            raise ValueError(f"Unknown method: {method}")

        return focal_length

    return None
