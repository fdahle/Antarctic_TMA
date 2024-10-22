""" georeference an orthophoto """

# Library imports
import copy

import cv2  # noqa
import numpy as np
from shapely.geometry import Polygon

# Local imports
import src.base.find_tie_points as ftp
import src.base.resize_image as rei
import src.base.rotate_image as ri
import src.base.rotate_points as rp
import src.display.display_images as di
import src.georef.snippets.apply_transform as at
import src.georef.snippets.calc_transform as ct
import src.load.load_satellite as ls

debug_show_tie_points = False


def georef_ortho(ortho: np.ndarray,
                 footprints: list, lst_aligned: list,
                 azimuth: int | None = None,
                 auto_rotate: bool = False,
                 rotation_step: int = 22.5,
                 max_size: int = 25000,  # in m
                 trim_image: bool = True,
                 min_nr_tps: int = 100,
                 tp_type: type = float,
                 save_path_ortho: str | None = None,
                 save_path_transform: str | None = None):
    """
    Georeference an ortho image created by Agisoft Metashape. Footprints and azimuth are used to
    get a rough bounding box of the ortho image. As the ortho image can be very large, it is
    divided into smaller tiles for geo-referencing.
    If no azimuth is given, the ortho image is rotated in steps of `rotation_step` degrees to find
    the best orientation.

    Args:
        ortho:
        footprints:
        lst_aligned:
        azimuth:
        auto_rotate:
        rotation_step:
        max_size:
        save_path:

    Returns:

    """

    # we need an ortho image with 2 dimensions
    if len(ortho.shape) == 3:
        ortho = copy.deepcopy(ortho)
        ortho = ortho[0, :, :]

    # init tie point detector
    tpd = ftp.TiePointDetector('lightglue', tp_type=tp_type)

    # Calculate the percentage removed from each side
    original_height_ortho, original_width_ortho = ortho.shape[:2]

    # trim the ortho
    if trim_image:
        trim_mask = ortho != 255  # Find non-white pixels
        trim_coords = np.argwhere(trim_mask)
        trim_y_min, trim_x_min = trim_coords.min(axis=0)
        trim_y_max, trim_x_max = trim_coords.max(axis=0) + 1  # Include the max pixel
        trimmed_ortho = ortho[trim_y_min:trim_y_max, trim_x_min:trim_x_max]

        left_trim_percentage = (trim_x_min / original_width_ortho) * 100
        right_trim_percentage = ((original_width_ortho - trim_x_max) / original_width_ortho) * 100
        top_trim_percentage = (trim_y_min / original_height_ortho) * 100
        bottom_trim_percentage = ((original_height_ortho - trim_y_max) / original_height_ortho) * 100

    else:
        trimmed_ortho = ortho
        trim_x_min = 0
        trim_y_min = 0
        left_trim_percentage = 0
        right_trim_percentage = 0
        top_trim_percentage = 0
        bottom_trim_percentage = 0

    # Filter aligned footprints
    aligned_footprints = [footprint for footprint, aligned in zip(footprints, lst_aligned) if aligned]

    print(f"Found {len(aligned_footprints)} aligned footprints out of {len(footprints)}")

    # Get the min and max for all footprints
    min_abs_x = min([footprint.bounds[0] for footprint in aligned_footprints])
    min_abs_y = min([footprint.bounds[1] for footprint in aligned_footprints])
    max_abs_x = max([footprint.bounds[2] for footprint in aligned_footprints])
    max_abs_y = max([footprint.bounds[3] for footprint in aligned_footprints])

    # Calculate the size of the bounding rectangle before trimming
    width_tmp = max_abs_x - min_abs_x
    height_tmp = max_abs_y - min_abs_y

    # Adjust the min and max coordinates based on the trim percentages
    min_abs_x = min_abs_x + (width_tmp * (left_trim_percentage / 100))
    max_abs_x = max_abs_x - (width_tmp * (right_trim_percentage / 100))
    min_abs_y = min_abs_y + (height_tmp * (top_trim_percentage / 100))
    max_abs_y = max_abs_y - (height_tmp * (bottom_trim_percentage / 100))

    # Calculate the size of the bounding rectangle
    width_abs_px = max_abs_x - min_abs_x
    height_abs_px = max_abs_y - min_abs_y

    if width_abs_px > max_size or height_abs_px > max_size:

        # check how often the width and height fit into the max size
        width_factor = width_abs_px // max_size
        height_factor = height_abs_px // max_size

        # factor should be at least 1
        width_factor = max(1, int(width_factor))
        height_factor = max(1, int(height_factor))

    # no tiling needed
    else:
        height_factor = 1
        width_factor = 1

    # divide by the factor to get width and height of the small tiles
    width_small = width_abs_px / width_factor
    height_small = height_abs_px / height_factor

    # 4 different cases for rotation values:
    # no azimuth but also no rotation (images are already aligned)
    if azimuth is None and auto_rotate is False:
        rotation_values = [0]
    # rotate the ortho in steps of rotation_step
    elif azimuth is None and auto_rotate is True:
        rotation_values = list(range(0, 360, int(rotation_step)))
    # azimuth is given, but we still want to rotate (for inaccurate azimuth)
    elif azimuth is not None and auto_rotate is True:

        rotation_values = [
            azimuth - rotation_step,
            azimuth - rotation_step / 2,
            azimuth,
            azimuth + rotation_step / 2,
            azimuth + rotation_step
        ]

        # assure all entries are between 0 and 360
        rotation_values = [x % 360 for x in rotation_values]

    # azimuth is given and we trust it
    elif azimuth is not None and auto_rotate is False:
        rotation_values = [azimuth]
    else:
        raise ValueError("Invalid combination of azimuth and auto_rotate")

    # get width and height of ortho
    ortho_width_small = int(trimmed_ortho.shape[1] / width_factor)
    ortho_height_small = int(trimmed_ortho.shape[0] / height_factor)

    # here we save the points
    points = []

    # iterate through all factors
    for h in range(height_factor):
        for w in range(width_factor):

            print(f" - Georef tile ({h}, {w})")

            # tile the ortho as well
            ortho_tile = trimmed_ortho[h * ortho_height_small:(h + 1) * ortho_height_small,
                                       w * ortho_width_small:(w + 1) * ortho_width_small]

            # get the new min and max values
            tile_min_x = min_abs_x + width_small * w
            tile_min_y = min_abs_y + height_small * h
            tile_max_x = tile_min_x + width_small
            tile_max_y = tile_min_y + height_small

            # Create the bounding rectangle
            tile_poly = Polygon([(tile_min_x, tile_min_y), (tile_min_x, tile_max_y),
                                 (tile_max_x, tile_max_y), (tile_max_x, tile_min_y)])

            # get satellite image for the bounding rectangle
            sat_tile, sat_transform_tile = ls.load_satellite(tile_poly)

            # init best values
            best_points = np.empty((0, 4))
            best_rot_matrix = None
            best_zoom_factors = None

            # iterate all rotations
            for rotation in rotation_values:
                print(f"  - try rotation of {rotation}")

                ortho_tile_rotated, rot_matrix = ri.rotate_image(ortho_tile, rotation,
                                                                 return_rot_matrix=True)

                # resize ortho to sat size
                zoom_factors = (sat_tile.shape[1] / ortho_tile_rotated.shape[0],
                                sat_tile.shape[2] / ortho_tile_rotated.shape[1])
                ortho_tile_resized = rei.resize_image(ortho_tile_rotated, (sat_tile.shape[1], sat_tile.shape[2]))

                # find tps between ortho and sat
                points_tile, _ = tpd.find_tie_points(sat_tile, ortho_tile_resized)

                # display the tie points
                if debug_show_tie_points:
                    style_config = {
                        "title": f"{points_tile.shape[0]} tie points found at rotation {rotation}",
                    }
                    di.display_images([sat_tile, ortho_tile_resized],
                                      tie_points=points_tile,
                                      style_config=style_config)

                # skip if no points were found
                if points_tile.shape[0] == 0:
                    print(f"  - No tie points found")
                    continue

                # remove outliers
                if points_tile.shape[0] > 4:
                    _, filtered_tile = cv2.findHomography(points_tile[:, 0:2],
                                                          points_tile[:, 2:4],
                                                          cv2.RANSAC, 5.0)
                    filtered_tile = filtered_tile.flatten()
                    points_tile = points_tile[filtered_tile == 0]

                # skip if no points were found
                if points_tile.shape[0] == 0:
                    print(f"  - No tie points found")
                    continue

                print(f"  - {points_tile.shape[0]} tie points found")

                # check if we have the best rotation
                if points_tile.shape[0] > best_points.shape[0]:
                    best_points = points_tile
                    best_rot_matrix = rot_matrix
                    best_zoom_factors = zoom_factors

            # no points found in this tile
            if best_rot_matrix is None:
                continue

            # get the best ortho
            points_tile = best_points
            rot_matrix = best_rot_matrix
            zoom_factors = best_zoom_factors

            # account for resizing
            points_tile[:, 2] = points_tile[:, 2] * (1 / zoom_factors[1])
            points_tile[:, 3] = points_tile[:, 3] * (1 / zoom_factors[0])

            # rotate points back
            points_tile[:, 2:] = rp.rotate_points(points_tile[:, 2:], rot_matrix, invert=True)

            # convert points to absolute values
            absolute_points_tile = np.array([sat_transform_tile * tuple(point) for point in points_tile[:, 0:2]])
            points_tile[:, 0:2] = absolute_points_tile

            # account for the tile offset
            points_tile[:, 2] = points_tile[:, 2] + w * ortho_width_small
            points_tile[:, 3] = points_tile[:, 3] + h * ortho_height_small

            # cast column 2 and 3 to int
            points_tile[:, 2] = points_tile[:, 2].astype(int)
            points_tile[:, 3] = points_tile[:, 3].astype(int)

            points.append(points_tile)

    # check if we found points
    if len(points) > 0:
        # convert list of points to numpy array
        points = np.vstack(points)
    else:
        points = np.empty((0, 4))

    # account for the trimming
    points[:, 2] = points[:, 2] + trim_x_min
    points[:, 3] = points[:, 3] + trim_y_min

    # check if we have enough points
    if points.shape[0] < min_nr_tps:
        return None, None

    # calculate the transform
    transform, residuals = ct.calc_transform(ortho, points,
                                             transform_method="rasterio",
                                             gdal_order=3)

    # apply the transform to the ortho
    if save_path_ortho is not None:
        at.apply_transform(trimmed_ortho, transform, save_path_ortho)

    # reshape the transform to a 3x3 matrix
    transform = np.reshape(transform, (3, 3))

    # calculate the absolute bounds
    corners = np.array([
        [0, 0],  # Top-left corner
        [ortho.shape[1], 0],  # Top-right corner
        [ortho.shape[1], ortho.shape[0]],  # Bottom-right corner
        [0, ortho.shape[0]],  # Bottom-left corner
    ])

    # Convert to homogeneous coordinates
    corners_homogeneous = np.hstack([corners, np.ones((corners.shape[0], 1))])

    # Apply the transformation matrix to the corner points
    transformed_corners = (transform @ corners_homogeneous.T).T

    # Convert back to 2D coordinates by dividing by the last coordinate
    transformed_corners = transformed_corners[:, :2] / transformed_corners[:, 2, np.newaxis]

    # Calculate the transformed geographic bounds
    min_trans_x = np.min(transformed_corners[:, 0])
    min_trans_y = np.min(transformed_corners[:, 1])
    max_trans_x = np.max(transformed_corners[:, 0])
    max_trans_y = np.max(transformed_corners[:, 1])

    transformed_bounds = (min_trans_x, min_trans_y, max_trans_x, max_trans_y)

    # check if the polygon of the transformed bounds overlaps with the polygon
    print("TODOOOOOOOOO-> enter check")

    # save the transform
    if save_path_transform is not None:
        np.savetxt(save_path_transform, transform, delimiter=',')

    return transform, transformed_bounds
