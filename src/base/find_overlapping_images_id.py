def find_overlapping_images_id(image_ids: list[str],
                               important_id: str | None = None,
                               max_id_range: int = 3):

    # check for file extensions (raise error if yes)
    for image_id in image_ids:
        if "." in image_id:
            raise ValueError("Image IDs should not contain file extensions")

    if max_id_range <= 0:
        raise ValueError("max_id_range must be a positive integer")

    if type(image_ids) is not list:
        raise ValueError("image_ids must be a list")

    if len(image_ids) < 2:
        raise ValueError("At least two image IDs are required to find overlaps")

    # Initialize the dictionary with all image_ids as keys and empty lists as values
    overlap_dict = {image_id: [] for image_id in image_ids}

    for i, id1 in enumerate(image_ids):

        # Skip if not the important_id
        if important_id and id1 != important_id:
            continue

        # Extract flight path, direction and number ID from the image ID
        flight_path1, direction1, num_id1 = id1[2:6], id1[6:9], int(id1[-4:])

        if important_id:
            start_idx = 0
        else:
            start_idx = i + 1

        for j in range(start_idx, len(image_ids)):

            id2 = image_ids[j]

            # Skip comparing with itself
            if id1 == id2:
                continue

            # Extract flight path and number ID from the image ID
            flight_path2, direction2, num_id2 = id2[2:6], id2[6:9], int(id2[-4:])

            # Check if the flight paths are the same and the number IDs are within the max_id_range
            if (flight_path1 == flight_path2 and direction1==direction2 and
                    abs(num_id1 - num_id2) <= max_id_range):
                overlap_dict[id1].append(id2)
                overlap_dict[id2].append(id1)

    # filter dict for important_id
    if important_id and important_id in overlap_dict.keys():
        overlap_dict = {important_id: overlap_dict[important_id]}

    return overlap_dict