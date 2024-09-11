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
import src.georef.snippets.apply_transform as at
import src.georef.snippets.calc_transform as ct
import src.load.load_satellite as ls


def georef_ortho(ortho, footprints, lst_aligned,
                 azimuth=None,
                 auto_rotate=False,
                 rotation_step=45,
                 max_size=25000,  # in m
                 save_path=None):


    # we need a ortho image with 2 dimensions
    if len(ortho.shape) == 3:
        ortho = copy.deepcopy(ortho)
        ortho = ortho[0, : , :]

    # init tie point detector
    tpd = ftp.TiePointDetector('lightglue')

    # Filter aligned footprints
    aligned_footprints = [footprint for footprint, aligned in zip(footprints, lst_aligned) if aligned]

    # Get the min and max for all footprints
    min_x = min([footprint.bounds[0] for footprint in aligned_footprints])
    min_y = min([footprint.bounds[1] for footprint in aligned_footprints])
    max_x = max([footprint.bounds[2] for footprint in aligned_footprints])
    max_y = max([footprint.bounds[3] for footprint in aligned_footprints])

    # Calculate the size of the bounding rectangle
    width_m = max_x - min_x
    height_m = max_y - min_y

    if width_m > max_size or height_m > max_size:

        # check how often the width and height fit into the max size
        width_factor = width_m // max_size
        height_factor = height_m // max_size

        # factor should be at least 1
        width_factor = max(1, int(width_factor))
        height_factor = max(1, int(height_factor))

    # no tiling needed
    else:
        height_factor = 1
        width_factor = 1

    # divide by the factor to get width and height of the small tiles
    width_small = width_m / width_factor
    height_small = height_m / height_factor

    # create rotation values to rotate dict
    if azimuth is None:

        # check if we want to auto rotate
        if auto_rotate is False:
            rotation_values = [0]
        else:
            # create a list of rotation values
            rotation_values = list(range(0, 360, rotation_step))
    else:
        rotation_values = [azimuth]

    # get width and height of ortho
    ortho_width_small = int(ortho.shape[1] / width_factor)
    ortho_height_small = int(ortho.shape[0] / height_factor)

    # here we save the points
    points = []

    # iterate through all factors
    for h in range(height_factor):
        for w in range(width_factor):

            print(f" - Georef tile ({h}, {w})")

            # tile the ortho as well
            ortho_tile = ortho[h * ortho_height_small:(h + 1) * ortho_height_small,
                               w * ortho_width_small:(w + 1) * ortho_width_small]

            # get the new min and max values
            tile_min_x = min_x + width_small * w
            tile_min_y = min_y + height_small * h
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

                import src.display.display_images as di
                #di.display_images([sat_tile, ortho_tile, ortho_tile_rotated, ortho_tile_resized])

                # find tps between ortho and sat
                points_tile, _ = tpd.find_tie_points(sat_tile, ortho_tile_resized)

                # skip if no points were found
                if points_tile.shape[0] == 0:
                    print(f"  - No tie points found")
                    continue

                # remove outliers
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
            point_tile = best_points
            rot_matrix = best_rot_matrix
            zoom_factors = best_zoom_factors

            # account for resizing
            points_tile[:, 2] = points_tile[:, 2] * (1 / zoom_factors[1])
            points_tile[:, 3] = points_tile[:, 3] * (1 / zoom_factors[0])

            # rotate points back
            point_tile[:, 2:] = rp.rotate_points(point_tile[:, 2:], rot_matrix, invert=True)

            # convert points to absolute values
            absolute_points_tile = np.array([sat_transform_tile * tuple(point)
                                             for point in points_tile[:, 0:2]])
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

    if points.shape[0] == 0:
        raise ValueError("No tie points found in all tiles")

    # calculate the transform
    transform, residuals = ct.calc_transform(ortho, points,
                                             transform_method="rasterio",
                                             gdal_order=3)

    if save_path is not None:
        at.apply_transform(ortho, transform, save_path)

    transform = np.reshape(transform, (3, 3))

    return transform
