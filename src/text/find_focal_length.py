# Package imports
import re

MIN_FOCAL_LENGTH = 140  # in mm
MAX_FOCAL_LENGTH = 160  # in mm
def find_focal_length(text):
    # split the text into their respective boxes
    text_per_box = text.split(";")

    for text_part in text_per_box:

        pattern = r'5\d\.\d{2,3}\d?'

        matches = re.findall(pattern, text_part)

    if len(matches) == 1:

        # convert the focal length to a float
        focal_length = float(matches[0])

        # increase focal length to 1xx.xx
        if focal_length < 100:
            focal_length = 100 + focal_length

        # check focal length
        if focal_length < MIN_FOCAL_LENGTH or focal_length > MAX_FOCAL_LENGTH:
            return None

        return focal_length

    return None