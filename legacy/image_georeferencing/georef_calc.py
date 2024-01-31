import copy
import numpy as np
import math

import display.display_images as di
import display.display_shapes as ds

import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.derive_new as dn

import base.load_image_from_file as liff
import base.remove_borders as rb
import base.rotate_image as ri

"""            if image_id == "CA164432V0029":
                sorted_points = np.array([bottom_left, bottom_right, top_right, top_left])
            elif image_id == "CA182332V0019":
                sorted_points = np.array([bottom_left, bottom_right, top_right, top_left])
            elif image_id == "CA207432V0157":
                sorted_points = np.array([top_right, top_left, bottom_left, bottom_right])
            elif image_id == "CA207332V0127":
                    sorted_points = np.array([bottom_left, bottom_right, top_right, top_left])
            elif image_id == "CA212432V0079":
                sorted_points = np.array([top_right, top_left, bottom_left, bottom_right])
"""


def georef_calc(image_id, path_fld, image=None, footprint=None, azimuth=None,
                transform_method="rasterio", transform_order=3,
                catch=True, verbose=False, pbar=None):
    """

    Args:
        image_id (String):  The id of the image we want to geo-reference
        path_fld (String): Where do we want to store the geo-referenced image
        footprint (Shapely-polygon, None): The polygon we are basing the geo-referencing on. If none, it is
         derived from the image position
        azimuth (int): The approximate rotation of the image
        transform_method (String, "rasterio"): Which library will be used for applying gcps ('rasterio' or 'gdal')
        transform_order (Int, 3): Which polynomial should be used for geo-referencing (only gdal)
        catch:
        pbar:

    Returns:

    """

    path_tiff = path_fld + "/" + image_id + ".tif"

    if footprint is None:
        point, footprint = dn.derive_new(image_id, mode="both", min_nr_of_images=2,
                                         polynomial_order=1,
                                         verbose=verbose, pbar=pbar)

        if point == "not_enough_images":
            return_tuple = (None, None, "not_enough_images", None, None)
            return return_tuple
        elif point == "wrong_overlap":
            return_tuple = (None, None, "wrong_overlap", None, None)
            return return_tuple

    if azimuth is None:
        sql_string = f"SELECT azimuth FROM images WHERE image_id='{image_id}'"
        import base.connect_to_db as ctd
        data = ctd.get_data_from_db(sql_string)
        azimuth = data['azimuth'].iloc[0]

    # check if we have an image
    if image is None:
        # load image
        image_with_borders = liff.load_image_from_file(image_id, catch=catch,
                                                       verbose=False, pbar=pbar)

        # get the border dimensions
        image = rb.remove_borders(image_with_borders, image_id=image_id,
                                  catch=catch, verbose=False, pbar=pbar)

    image, corners = ri.rotate_image(image, azimuth, return_corners=True)

    # get corners from footprint
    tps_abs = np.asarray(list(footprint.exterior.coords)[:-1])  # excluding the repeated last point

    if tps_abs.shape[0] != 4:
        return None, None, None, None, None

    def sort_corners(coords, inverse=False):

        # If coords is a numpy array, convert it to a list of tuples
        if isinstance(coords, np.ndarray):
            coords = [tuple(point) for point in coords]

        x_coords = [p[0] for p in coords]
        y_coords = [p[1] for p in coords]
        centroid_x = sum(x_coords) / len(coords)
        centroid_y = sum(y_coords) / len(coords)

        # Function to calculate the polar angle with respect to the centroid
        def compute_angle(point):
            return math.atan2(point[1] - centroid_y, point[0] - centroid_x)

        # Sort points by polar angle
        sorted_points = sorted(coords, key=compute_angle)

        # Find the top-left point
        # If inverse is True, then higher y-values are considered top
        # If inverse is False, then smaller y-values are considered top
        if inverse:
            top_left = min(sorted_points, key=lambda p: (p[0], -p[1]))
        else:
            top_left = min(sorted_points, key=lambda p: (p[1], p[0]))

        # Find the top-left point's index
        top_left_index = sorted_points.index(top_left)

        # Reorder the list so that the top-left point is the first point
        sorted_points = sorted_points[top_left_index:] + sorted_points[:top_left_index]

        # Adjusting the ordering to always return [TL, TR, BR, BL]
        if inverse:
            # For inverse, the current order is [TL, BL, BR, TR]
            sorted_points = [sorted_points[i] for i in [0, 3, 2, 1]]

        return sorted_points

    # sort tps
    tps_img = sort_corners(corners, inverse=True)

    tps_img = np.asarray(tps_img)
    tps_abs = np.asarray(tps_abs)

    # create merged tie-point array
    tps = np.concatenate((tps_abs, tps_img), axis=1)

    transform, residuals = ag.apply_gcps(path_tiff, image, tps,
                                         transform_method, transform_order,
                                         return_error=True, save_image=True,
                                         catch=catch)

    return footprint, transform, "georeferenced", None, residuals
