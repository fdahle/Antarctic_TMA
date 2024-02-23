from typing import Dict, Optional
from shapely.geometry import Polygon


def find_overlapping_images(
        image_ids: list[str],
        footprints: Optional[list[Polygon]] = None,
        important_id: Optional[str] = None,
        max_id_range: int = 2,
        min_overlap: Optional[float] = None,
        working_modes: Optional[list[str]] = None) -> \
        Dict[str, list[str]]:
    """
    Finds overlapping images based on their IDs or geographical footprints.
    Args:
        image_ids: A list of image identifiers.
        footprints: A list of shapely Polygon objects representing image footprints.
        important_id: A specific image ID to check overlaps for. If set, the function only checks
            overlaps for this ID.
        max_id_range: The maximum ID range for considering two images from the same
            flight path as overlapping.
        min_overlap: The minimum overlap percentage required to consider two
            footprints as overlapping.
        working_modes: The modes of operation. Can include "ids" for ID-based overlap,
            "footprints" for geographic overlap, or both.
    Returns:
        A dictionary where each key is an image ID, and the value is a list of IDs that
            overlap with it.
    Raises:
        ValueError: If required parameters are not provided or invalid.
    """

    if working_modes is None:
        working_modes = ["ids", "footprints"]

    if "ids" in working_modes and max_id_range <= 0:
        raise ValueError("max_id_range must be a positive integer")

    if "footprints" in working_modes:
        if footprints is None:
            raise ValueError("footprints must be provided if 'footprints' is in working_modes")
        if min_overlap is not None and (min_overlap < 0 or min_overlap > 1):
            raise ValueError("min_overlap must be a float between 0 and 1")

    # Initialize the dictionary with all image_ids as keys and empty lists as values
    overlap_dict = {image_id: [] for image_id in image_ids}

    if "ids" in working_modes:
        for i, id1 in enumerate(image_ids):

            # Skip if not the important_id
            if important_id and id1 != important_id:
                continue

            # Extract flight path and number ID from the image ID
            flight_path1, num_id1 = id1[2:6], int(id1[-4:])

            for j in range(i + 1, len(footprints)):

                id2 = image_ids[j]

                # Skip comparing with itself
                if id1 == id2:
                    continue

                # Extract flight path and number ID from the image ID
                flight_path2, num_id2 = id2[2:6], int(id2[-4:])

                # Check if the flight paths are the same and the number IDs are within the max_id_range
                if flight_path1 == flight_path2 and abs(num_id1 - num_id2) <= max_id_range:
                    overlap_dict[id1].append(id2)

    if "footprints" in working_modes:
        # Iterate through each polygon and compare it with the subsequent polygons only
        for i, poly1 in enumerate(footprints):

            # Only process if it's the important_id
            if important_id and image_ids[i] != important_id:
                continue

            for j in range(i + 1, len(footprints)):
                poly2 = footprints[j]

                # Check if the IDs are already in the overlap_dict
                if image_ids[j] in overlap_dict[image_ids[i]]:
                    continue

                # Check if polygons intersect
                if poly1.intersects(poly2):

                    if min_overlap is not None:

                        # Calculate overlap percentage
                        overlap_percentage = (poly1.intersection(poly2).area / poly1.union(poly2).area) * 100

                        # they overlap more than the min_overlap
                        if overlap_percentage >= min_overlap:
                            overlap_dict[image_ids[i]].append(image_ids[j])
                            overlap_dict[image_ids[j]].append(image_ids[i])

                    else:
                        # If no min_overlap specified, just add the overlapping IDs
                        overlap_dict[image_ids[i]].append(image_ids[j])
                        overlap_dict[image_ids[j]].append(image_ids[i])

    return overlap_dict
