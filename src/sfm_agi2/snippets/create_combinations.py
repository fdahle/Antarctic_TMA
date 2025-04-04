def create_combinations(image_ids,
                        matching_method,
                        footprint_dict=None,
                        max_step_range=10,
                        min_overlap=0.5):

    combinations = {}

    if matching_method in ["overlap", "combined"] and footprint_dict is None:
        raise ValueError("Footprint dictionary is required for overlap matching")

    if matching_method == "all":

        # create a dict with all combinations
        for i, img_id in enumerate(image_ids):
            combinations[img_id] = image_ids[i + 1:]

    elif matching_method == "sequential":

        # create image groups based on flight and direction
        image_groups = {}
        for img_id in image_ids:
            flight = img_id[2:6]
            direction = img_id[6:9]
            number = int(img_id[-4:])
            key = (flight, direction)
            image_groups.setdefault(key, []).append((number, img_id))

        for _, images in image_groups.items():
            images.sort()
            numbers, ids = zip(*images)
            for i in range(len(ids)):
                current_id = ids[i]
                for step in range(1, step_range + 1):
                    if i + step < len(ids):
                        combinations.setdefault(current_id, []).append(ids[i + step])


    elif matching_method == "overlap":

        footprints_lst = [footprint_dict[image_id] for image_id in image_ids]
        overlap_dict = foi.find_overlapping_images_geom(image_ids, footprints_lst, min_overlap=min_overlap)

        for img_id, overlap_lst in overlap_dict.items():
            for overlap_id in overlap_lst:
                if overlap_id not in combinations and img_id != overlap_id:
                    combinations.setdefault(img_id, []).append(overlap_id)

    elif matching_method == "combined":
        pass

    else:
        raise ValueError("Unknown matching method")

    return combinations