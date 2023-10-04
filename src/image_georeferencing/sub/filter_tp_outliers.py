import copy
import cv2
import numpy as np

import base.print_v as p


def filter_tp_outliers(points, confidences=None,
                       use_ransac=True, ransac_threshold=10,
                       use_angle=False, angle_threshold=45,
                       verbose=False, pbar=None):
    """
    filter_tp_outliers(points, confidences, use_ransac, ransac_threshold, use_angle, angle_threshold, verbose, pbar):
    This function filters the tie-points between two images and keeps (hopefully) only the useful tie-points. Filtering
    can be applied with ransac or with angles (The average angle of all lines between the tie-points is calculated and
    used as the value for filtering).
    Args:
        points (np-array): The tie-points between images as a numpy-array (X, 4), with x1, y1, x2, y2
        confidences (list, None): An optional list of confidences of the tie-points that also can be filtered
        use_ransac (Boolean, true): If true, ransac is used for filtering
        ransac_threshold (int, 10): the limit for outliers for ransac (as smaller, as more strict is filtering)
        use_angle (Boolean, false): Should the tie-points be filtered based on their angle between two images
        angle_threshold (int, 45): The maximum allowed deviation of the angles of the lines of the tie-points
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        points (np-array): The filtered tie-points
        confidences (list): The filtered confidences
    """

    points = copy.deepcopy(points)
    if confidences is not None:
        confidences = copy.deepcopy(confidences)

    def filter_by_angle(_points, _angle_threshold):
        """
        Filter tie-points based on the angle of their displacement vectors.

        :param _points: np.array of shape (x, 4) where columns represent [x1, y1, x2, y2].
        :param _angle_threshold: maximum allowed angular deviation from the median angle.

        :return: A boolean mask indicating which tie-points have consistent angles.
        """

        # we only need to filter by angle if we have more than 1 point
        if _points.shape[0] <= 1:
            consistent_mask = np.zeros((_points.shape[0], 1))
        else:
            min_x_1 = np.amin(_points[:, 0])
            max_x_1 = np.amax(_points[:, 0])
            min_x_2 = np.amin(_points[:, 2])
            max_x_2 = np.amax(_points[:, 2])

            min_y_1 = np.amin(_points[:, 1])
            max_y_1 = np.amax(_points[:, 1])
            min_y_2 = np.amin(_points[:, 3])
            max_y_2 = np.amax(_points[:, 3])

            width_1 = max_x_1 - min_x_1
            height_1 = max_y_1 - min_y_1

            width_2 = max_x_2 - min_x_2
            height_2 = max_y_2 - min_y_2

            difference_width = abs(width_1 - width_2) / max(width_1, width_2)
            difference_height = abs(height_1 - height_2) / max(height_1, height_2)

            # only change temporary points
            copied_points = copy.deepcopy(_points)

            copied_points[:, 2] = copied_points[:, 2] * (1 - difference_width)
            copied_points[:, 3] = copied_points[:, 3] * (1 - difference_height)

            # Calculate displacement vectors for all tie-points
            displacements = copied_points[:, 2:] - copied_points[:, :2]

            debug_show_displacements = False
            if debug_show_displacements:
                import matplotlib.pyplot as plt
                # Get the start points of your vectors (points from image 1)
                _x = copied_points[:, 0]
                _y = copied_points[:, 1]

                # Get the displacements
                _u = displacements[:, 0]
                _v = displacements[:, 1]

                # Create a quiver plot
                plt.figure(figsize=(10, 10))
                plt.quiver(_x, _y, _u, _v, angles='xy', scale_units='xy', scale=1, color='r')

                # Setting the aspect ratio to 'equal' for accurate representation
                plt.gca().set_aspect('equal', adjustable='box')
                plt.title("Displacement Vectors")
                plt.show()

            # Calculate angles of the displacements (in degrees)
            angles = np.degrees(np.arctan2(displacements[:, 1], displacements[:, 0]))

            # Calculate the median angle
            median_angle = np.median(angles)

            # Determine which tie-points have angles within the threshold of the median
            lower_bound = median_angle - _angle_threshold
            upper_bound = median_angle + _angle_threshold
            consistent_mask = (angles >= lower_bound) & (angles <= upper_bound)

            consistent_mask = np.invert(consistent_mask)

        return consistent_mask

    if use_angle:

        mask = filter_by_angle(points, angle_threshold)

        points = points[mask.flatten() == 0]
        if confidences is not None:
            confidences = confidences[mask.flatten() == 0]

        p.print_v(f"{np.count_nonzero(mask.flatten())} entries are removed as outliers (using angles)",
                  verbose=verbose, pbar=pbar)

    if use_ransac and points.shape[0] >= 4:
        _, mask = cv2.findHomography(points[:, 0:2], points[:, 2:4], cv2.RANSAC, ransac_threshold)
        points = points[mask.flatten() == 0]
        if confidences is not None:
            confidences = confidences[mask.flatten() == 0]

        p.print_v(f"{np.count_nonzero(mask.flatten())} entries are removed as outliers (using ransac)",
                  verbose=verbose, pbar=pbar)

    if confidences is None:
        return points
    else:
        return points, confidences


if __name__ == "__main__":

    image_id = "CA183532V0040"

    import base.load_image_from_file as liff

    image_with_borders = liff.load_image_from_file(image_id)

    sat_min_x = -2110819
    sat_max_x = -2100234
    sat_min_y = 764121
    sat_max_y = 776314
    sat_bounds = [sat_min_x, sat_min_y, sat_max_x, sat_max_y]

    import image_georeferencing.sub.load_satellite_data as lsd

    sat = lsd.load_satellite_data(sat_bounds, verbose=True)

    import base.connect_to_db as ctd

    sql_string = "SELECT image_id, height, focal_length, complexity, " \
                 "ST_AsText(footprint_approx) AS footprint_approx, " \
                 "ST_AsText(footprint) AS footprint, " \
                 "footprint_type AS footprint_type " \
                 f"FROM images_extracted WHERE image_id ='{image_id}'"
    data_extracted = ctd.get_data_from_db(sql_string, catch=False, verbose=True)

    import shapely.wkt

    footprint_approx = shapely.wkt.loads(data_extracted['footprint_approx'].iloc[0])

    import image_georeferencing.sub.adjust_image_resolution as air

    image_adjusted = air.adjust_image_resolution(sat,
                                                 sat_bounds,
                                                 image_with_borders,
                                                 footprint_approx.bounds)

    import base.remove_borders as rb

    _, edge_dims = rb.remove_borders(image_with_borders, image_id=image_id,
                                     return_edge_dims=True)

    # adapt the border dimension to account for smaller image
    change_y = image_with_borders.shape[0] / image_adjusted.shape[0]
    change_x = image_with_borders.shape[1] / image_adjusted.shape[1]
    if change_y > 0 or change_x > 0:
        change_y = 1 / change_y
        change_x = 1 / change_x
    edge_dims[0] = int(edge_dims[0] * change_x)
    edge_dims[1] = int(edge_dims[1] * change_x)
    edge_dims[2] = int(edge_dims[2] * change_y)
    edge_dims[3] = int(edge_dims[3] * change_y)

    image = image_adjusted[edge_dims[2]:edge_dims[3], edge_dims[0]:edge_dims[1]]

    import image_georeferencing.sub.enhance_image as eh

    enhancement_min = 0.02
    enhancement_max = 0.98

    image = eh.enhance_image(image, scale=(enhancement_min, enhancement_max))

    sql_string = f"SELECT * FROM images WHERE image_id='{image_id}'"
    data = ctd.get_data_from_db(sql_string)

    angle = data["azimuth"].iloc[0]

    import base.rotate_image as ri

    image = ri.rotate_image(image, angle, start_angle=0)

    import image_tie_points.find_tie_points as ftp

    tps, conf = ftp.find_tie_points(sat, image, min_threshold=0,
                                    filter_outliers=False,
                                    verbose=True, matching_method="LightGlue")

    import display.display_tiepoints as dt

    dt.display_tiepoints([sat, image], tps, conf)

    tps, conf = filter_tp_outliers(tps, conf, use_angle=True, verbose=True)

    dt.display_tiepoints([sat, image], tps, conf, title="Final tps")
