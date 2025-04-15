"""Create image pair combinations for tie-point matching"""

from collections import defaultdict

import src.base.find_overlapping_images_geom as foig

def create_combinations(image_ids: list[str],
                        matching_method: str,
                        footprint_dict=None,
                        step_range: int = 3,
                        max_step_difference:int = 10,
                        min_overlap: float= 0.5,
                        pairwise: bool = False) -> (
        dict[str, list[str]] | list[tuple[int, int]]):
    """
    The function returns a dictionary mapping each image ID to a list of
    other image IDs that should be matched with it. Each pair is only included
    once (i.e., if image A is paired with image B, image B will not be paired
    again with image A).

    Args:
        image_ids (list[str]): List of image IDs to be considered for matching.
        matching_method (str): Matching strategy to use. Options are:
            - "all": Match all unique pairs.
            - "sequential": Match within the same flight and direction using
                `step_range`.
            - "overlap": Match images with overlapping footprints.
            - "combined": Combines "sequential", "overlap", and L-V-R
                triplet matching.
        footprint_dict (dict[str, np.ndarray], optional): Dictionary mapping
            image IDs to footprint polygons (required for "overlap" and "combined").
        step_range (int, optional): Step range for sequential matching (default is 3).
        max_step_difference (int, optional): Max allowed difference in
            image numbering for filtering pairs (default is 10).
        min_overlap (float, optional): Minimum overlap threshold for
            footprint matching (default is 0.5).
        pairwise (bool, optional): If True, returns a list of tuples with
            indices of the image pairs instead of a dictionary (default is False).
    Returns:
        dict[str, list[str]]: Dictionary where keys are image IDs and values
        are lists of other image IDs to match with. Each pair is only included
        once and respects the filtering and matching logic defined by the
        chosen method.

    Raises:
        ValueError: If an unknown matching method is passed or if footprint_dict is
        missing when required.
    """

    if matching_method in ["overlap", "combined"] and footprint_dict is None:
        raise ValueError("Footprint dictionary is required for overlap matching")

    combinations = defaultdict(set)

    if matching_method == "all":

        # create a dict with all combinations
        for i, img_id in enumerate(image_ids):
            for j in range(i + 1, len(image_ids)):
                combinations[img_id].add(image_ids[j])

    elif matching_method == "sequential":

        # Create a dictionary to group images by (flight, direction)
        image_groups = defaultdict(list)

        # Group images by flight and direction
        for img_id in image_ids:
            flight = img_id[2:6]  # Extract flight number
            direction = img_id[6:9]  # Extract direction
            number = int(img_id[-4:])  # Convert last 4 digits to integer for easy comparisons
            key = (flight, direction)
            image_groups[key].append((number, img_id))

        # For each (flight, direction) group, create sequential pairs based on step range
        for _, images in image_groups.items():
            # Sort images by their sequential number
            images.sort()
            numbers, ids = zip(*images)  # Separate numbers and IDs

            for i in range(len(ids)):
                for step in range(1, step_range + 1):
                    if i + step < len(ids):
                        combinations[ids[i]].add(ids[i + step])

    elif matching_method == "overlap":

        footprints_lst = [footprint_dict[image_id] for image_id in image_ids]
        overlap_dict = foig.find_overlapping_images_geom(image_ids, footprints_lst, min_overlap=min_overlap)

        for img_id, overlaps in overlap_dict.items():
            for other in overlaps:
                if img_id < other:
                    combinations[img_id].add(other)
                elif other < img_id and other not in combinations:
                    combinations[other].add(img_id)

    elif matching_method == "combined":

        # get matching with the other methods
        sequential = create_combinations(image_ids, "sequential",
                                         step_range=step_range)
        overlap = create_combinations(image_ids, "overlap",
                                      footprint_dict=footprint_dict,
                                      min_overlap=min_overlap)

        # Triplet combinations
        triplets = defaultdict(set)
        for img_id in image_ids:
            flight = img_id[2:6]
            direction = img_id[6:9]
            number = img_id[-4:]

            img_l = f"CA{flight}31L{number}"
            img_v = f"CA{flight}32V{number}"
            img_r = f"CA{flight}33R{number}"

            if direction == "31L" and img_v in image_ids:
                triplets[img_id].add(img_v)
            elif direction == "32V":
                if img_l in image_ids:
                    triplets[img_id].add(img_l)
                if img_r in image_ids:
                    triplets[img_id].add(img_r)
            elif direction == "33R" and img_v in image_ids:
                triplets[img_id].add(img_v)

        # Merge all methods
        for combo_dict in [sequential, overlap, triplets]:
            for key, values in combo_dict.items():
                for val in values:
                    if key < val:
                        combinations[key].add(val)
                    elif val < key and val not in combinations:
                        combinations[val].add(key)
    else:
        raise ValueError("Unknown matching method")

    # Apply max_step_difference filtering
    filtered_combinations = defaultdict(list)
    for img1, targets in combinations.items():
        f1 = img1[2:6]
        d1 = img1[6:9]
        n1 = int(img1[-4:])

        for img2 in targets:
            f2 = img2[2:6]
            d2 = img2[6:9]
            n2 = int(img2[-4:])

            if f1 == f2 and d1 == d2 and abs(n1 - n2) > max_step_difference:
                continue
            filtered_combinations[img1].append(img2)

    # If pairwise=True, convert to list of (index, index) tuples
    if pairwise:
        image_ids_sorted = sorted(image_ids)
        image_index = {img_id: i for i, img_id in enumerate(image_ids_sorted)}
        pairs = []

        for img1, targets in filtered_combinations.items():
            for img2 in targets:
                idx1 = image_index[img1]
                idx2 = image_index[img2]
                pair = (min(idx1, idx2), max(idx1, idx2))
                pairs.append(pair)

        # Remove duplicates and return sorted list
        return sorted(set(pairs))

    return filtered_combinations