import copy
import numpy as np
import math

import display.display_images as di
import display.display_shapes as ds

import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.derive_image_position as dip

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

    if footprint is None or azimuth is None:
        _, footprint, regression_line = dip.derive_image_position(image_id, polygon_mode="exact",
                                                                  return_line=True, verbose=True, pbar=pbar)

        if footprint is None:
            return_tuple = (None, None, "not_enough_images", None, None)
            return return_tuple

        # we need the angle of the regression line
        x1, y1 = regression_line.coords[0]
        x2, y2 = regression_line.coords[1]
        angle_rad = math.atan2(x2 - x1, y2 - y1)
        azimuth = math.degrees(angle_rad)

    print(image_id, azimuth)

    # check if we have an image
    if image is None:

        # load image
        image_with_borders = liff.load_image_from_file(image_id, catch=catch,
                                                       verbose=False, pbar=pbar)

        # get the border dimensions
        image = rb.remove_borders(image_with_borders, image_id=image_id,
                                  catch=catch, verbose=False, pbar=pbar)

        img = copy.deepcopy(image)

        image, corners = ri.rotate_image(image, azimuth, return_corners=True)


        #display.display_images.display_images([img, image])

    def sort_corners(coords, inverse=False):

        # If coords is a numpy array, convert it to a list of tuples
        if isinstance(coords, np.ndarray):
            coords = [tuple(point) for point in coords]

        print(coords)

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

        print("TOLE", top_left)

        # Find the top-left point's index
        top_left_index = sorted_points.index(top_left)

        # Reorder the list so that the top-left point is the first point
        sorted_points = sorted_points[top_left_index:] + sorted_points[:top_left_index]

        # Adjusting the ordering to always return [TL, TR, BR, BL]
        if inverse:
            # For inverse, the current order is [TL, BL, BR, TR]
            sorted_points = [sorted_points[i] for i in [0, 3, 2, 1]]

        return sorted_points

    # get corners from footprint
    tps_abs = np.asarray(list(footprint.exterior.coords)[:-1])  # excluding the repeated last point


    inverse = True
    # sort tps
    tps_img = sort_corners(corners, inverse=inverse)
    tps_abs = sort_corners(tps_abs, inverse=inverse)

    from shapely.geometry import Polygon
    poly_img = Polygon(tps_img)

    from shapely.geometry import Point
    import display.display_shapes as ds
    tps_img_shapes = [Point(coord) for coord in tps_img]
    #ds.display_shapes(tps_img_shapes, subtitles=["TL", "TR", "BR", "BL"])

    tps_rel = copy.deepcopy(np.asarray(tps_abs))
    tps_rel[:,0] = tps_rel[:, 0] - np.amin(tps_rel[:,0])
    tps_rel[:,1] = tps_rel[:, 1] - np.amin(tps_rel[:,1])

    poly_rel = Polygon(tps_rel)


    from shapely.geometry import Point
    points_img = [Point(xy) for xy in tps_img]
    points_rel = [Point(xy) for xy in tps_rel]

    titles = ["TL", "TR", "BR", "BL"]

    #ds.display_shapes([poly_img, poly_rel, points_img, points_rel], subtitles=["","",titles, titles])
    #exit()

    tps_img = [tps_img[-1]] + tps_img[:-1]

    tps_img = np.asarray(tps_img)
    tps_abs = np.asarray(tps_abs)

    # create merged tie-point array
    tps = np.concatenate((tps_abs, tps_img), axis=1)


    transform, residuals = ag.apply_gcps(path_tiff, image, tps,
                                         transform_method, transform_order,
                                         return_error=True, save_image=True,
                                         catch=catch)

    import display.display_shapes as ds
    #ds.display_shapes([footprint])

    return footprint, transform, "georeferenced", None, residuals

