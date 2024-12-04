"""find overlapping images based on their IDs or geographical footprints."""

# Library imports
import geopandas as gpd
from typing import Dict, Optional
from shapely.geometry import Polygon

import src.display.display_shapes as ds

debug_show_overlap = False
debug_only_matches = False

def find_overlapping_images_geom(
        image_ids: list[str],
        footprints: Optional[list[Polygon]] = None,
        important_id: Optional[str] = None,
        min_overlap: Optional[float] = 0.0) -> \
        Dict[str, list[str]]:
    """
    Finds overlapping images based on geographical footprints.
    Args:
        image_ids: A list of image identifiers in the format CAXXXX32VXXXX.
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

    # check for file extensions (raise error if yes)
    for image_id in image_ids:
        if "." in image_id:
            raise ValueError("Image IDs should not contain file extensions")

    if type(image_ids) is not list:
        raise ValueError("image_ids must be a list")

    if footprints is None:
        raise ValueError("footprints must be provided if 'footprints' is in working_modes")

    if isinstance(footprints, gpd.GeoSeries):
        footprints_list = list(footprints.geometry)
    else:
        footprints_list = footprints

    # Initialize the dictionary with all image_ids as keys and empty lists as values
    overlap_dict = {image_id: [] for image_id in image_ids}


    # Iterate through each polygon and compare it with the subsequent polygons only
    for i, poly1 in enumerate(footprints_list):

        # Only process if it's the important_id
        if important_id and image_ids[i] != important_id:
            continue

        for j in range(i + 1, len(footprints_list)):
            poly2 = footprints_list[j]

            # Check if the IDs are already in the overlap_dict
            if image_ids[j] in overlap_dict[image_ids[i]]:
                continue

            # Check if polygons intersect
            if poly1.intersects(poly2):

                # Calculate overlap percentage
                intersection_area = poly1.intersection(poly2).area
                smaller_area = min(poly1.area, poly2.area)
                overlap_percentage = intersection_area / smaller_area

                if overlap_percentage >= min_overlap:
                    overlap_dict[image_ids[i]].append(image_ids[j])
                    overlap_dict[image_ids[j]].append(image_ids[i])

                if debug_show_overlap:
                    style_config = {
                        "title": f"Overlap between {image_ids[i]} and {image_ids[j]}: "
                                 f"{round(overlap_percentage, 3)} overlap",
                        "colors": ["red", "blue"]
                    }

                    ds.display_shapes([poly1, poly2], style_config=style_config)
            else:
                if debug_show_overlap and debug_only_matches is False:
                    style_config = {
                        "title": f"Overlap between {image_ids[i]} and {image_ids[j]}: no overlap",
                        "colors": ["red", "blue"]
                    }

                    ds.display_shapes([poly1, poly2], style_config=style_config)

    # remove potential duplicates
    for key in overlap_dict.keys():
        overlap_dict[key] = list(set(overlap_dict[key]))

    # filter dict for important_id
    if important_id and important_id in overlap_dict.keys():
        overlap_dict = {important_id: overlap_dict[important_id]}

    return overlap_dict
