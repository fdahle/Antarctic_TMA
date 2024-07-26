"""generate a mask for an image"""

# Library imports
import numpy as np

# Local imports
import src.base.connect_to_database as ctd


def create_mask(image: np.ndarray,
                fid_marks: str | tuple[int, int] | None = None,
                ignore_boxes: list[tuple[int, int, int, int]] | None = None,
                use_default_fiducials: bool = False,
                default_fid_position: int = 500,
                min_border_width: int = None,
                use_database: bool = False,
                image_id=None,
                ) -> np.ndarray:
    """
    This function generates a mask for an image based on provided fiducial marks (everything outside
    these marks is masked). If no marks are provided, it is possible to use default positions.
    Additionally, specific boxes within the image can be ignored by setting them to zero in the mask.
    Masked areas are set to 0, unmasked areas are set to 1.

    3 ## 7 ## 2
    #         #
    5 # PPA # 6
    #         #
    1 ## 8 ## 4

    Args:
        image: A numpy array representing the image to mask.
        fid_marks: An optional dictionary of fiducial marks with keys as string identifiers
            and values as tuples representing positions (x, y). If None, default positions are used.
        ignore_boxes: An optional list of boxes to ignore, each specified as a tuple of four integers
            (x1, y1, x2, y2) representing the top left (x1, y1) and bottom right (x2, y2) corners.
        use_default_fiducials: A boolean flag to use default fiducial marks if no fiducial marks are provided.
        default_fid_position: An optional integer defining the default position for fiducial marks
            if none are provided. Defaults to 500.
        min_border_width: An optional integer defining the minimum border width to apply to the mask.
        use_database: A boolean flag indicating whether to fetch fiducial marks and ignore boxes
            from a database.
        image_id: An optional identifier for the image to fetch fiducial marks and ignore boxes
            from the database if `use_database` is True.
    Returns:
        A numpy array representing the masked image, where regions outside the specified fiducial
        marks and ignore boxes are set to zero.
    Raises:
        ValueError: If `use_default_fiducials` is False and any required fiducial mark is missing.
                    If `use_database` is True and `image_id` is None.
    """

    # replace the values with database values
    if use_database:

        conn = ctd.establish_connection()

        if image_id is None:
            raise ValueError("Image ID is required when using the database.")

        # get mask data for img
        sql_string_fid_marks = f"SELECT * FROM images_fid_points WHERE image_id='{image_id}'"
        data_fid_marks = ctd.execute_sql(sql_string_fid_marks, conn)

        sql_string_extracted = f"SELECT * FROM images_extracted WHERE image_id='{image_id}'"
        data_extracted = ctd.execute_sql(sql_string_extracted, conn)

        # Get the fid marks for the specific image_id
        fid_marks_row = data_fid_marks.loc[data_fid_marks['image_id'] == image_id].squeeze()

        # Create fid mark dict using dictionary comprehension
        fid_marks = {str(i): (fid_marks_row[f'fid_mark_{i}_x'], fid_marks_row[f'fid_mark_{i}_y']) for i in range(1, 5)}

        # get the text boxes of the image
        text_string = data_extracted.loc[data_extracted['image_id'] == image_id]['text_bbox'].iloc[0]

        if len(text_string) > 0 and "[" not in text_string:
            text_string = "[" + text_string + "]"

        # create list for parts to ignor
        ignore_boxes = [list(group) for group in eval(text_string.replace(";", ","))]

    # create base mask
    mask = np.ones_like(image)

    # Define default positions
    default_positions = {
        "1": (default_fid_position, mask.shape[0] - default_fid_position),
        "2": (mask.shape[1] - default_fid_position, default_fid_position),
        "3": (default_fid_position, default_fid_position),
        "4": (mask.shape[1] - default_fid_position, mask.shape[0] - default_fid_position),
    }

    # Initialize fid_marks if None, or fill in missing/default for existing keys
    if fid_marks is None:
        if use_default_fiducials:
            fid_marks = default_positions
        else:
            raise ValueError("Fiducial marks are required when use_default_fiducials is False.")
    else:
        if not use_default_fiducials:

            # check for missing entries
            missing_marks = [key for key in default_positions if key not in fid_marks]
            if missing_marks:
                raise ValueError(f"Missing fiducial marks: {missing_marks}.")

            # check for None values in the entries
            for key, value_tuple in fid_marks.items():
                # Check if 'None' is in the tuple
                if None in value_tuple:
                    raise ValueError("There are invalid entries in the fid-marks")

        # fill missing keys with default values
        for key, default_position in default_positions.items():
            fid_marks.setdefault(key, default_position)

        # fill none values with default values
        for key, default_position in default_positions.items():
            if None in fid_marks[key]:
                fid_marks[key] = default_position

    # get the min and max x/y values from the fid marks
    min_x = int(max(fid_marks["3"][0], fid_marks["1"][0]))
    max_x = int(min(fid_marks["2"][0], fid_marks["4"][0]))
    min_y = int(max(fid_marks["3"][1], fid_marks["2"][1]))
    max_y = int(min(fid_marks["1"][1], fid_marks["4"][1]))

    # Apply the min_border_width if specified
    if min_border_width is not None:
        min_x = max(min_x, min_border_width)
        max_x = min(max_x, mask.shape[1] - min_border_width)
        min_y = max(min_y, min_border_width)
        max_y = min(max_y, mask.shape[0] - min_border_width)

    # mask the borders
    # Set top and bottom regions to 0
    mask[:min_y, :] = 0
    mask[max_y:, :] = 0

    # Set left and right regions to 0
    mask[:, :min_x] = 0
    mask[:, max_x:] = 0

    # Mask the ignore_boxes if any
    if ignore_boxes is not None:
        for box in ignore_boxes:
            x1, y1, x2, y2 = box
            mask[int(y1):int(y2), int(x1):int(x2)] = 0

    return mask
