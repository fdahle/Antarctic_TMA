""" georeference an orthophoto for sfm"""

# Library imports
import copy

import numpy as np

# Local imports
import src.base.find_tie_points as ftp
import src.base.resize_max as rm
import src.base.rotate_image as ri
import src.base.rotate_points as rp
import src.display.display_images as di
import src.georef.snippets.apply_transform as at
import src.georef.snippets.calc_transform as ct
import src.load.load_satellite as ls
import src.export.export_geometry as eg

debug_min_tps = 25

# debug settings
debug_show_patch_tie_points = False
debug_show_rotated_tie_points = False
debug_show_final_tie_points = False

debug_save_sat = True
debug_save_footprints = True

def georef_ortho(ortho: np.ndarray,
                 image_ids: list[str],
                 footprints: list,
                 lst_aligned: list[bool],
                 sat_bounds = None,
                 azimuth: int | None = None,
                 auto_rotate: bool = False,
                 rotation_step: int = 5,
                 max_size: int = 25000,  # in m
                 trim_image: bool = True,
                 min_nr_tps: int = 100,
                 min_conf: float = 0.5,
                 start_conf: float = 0.9,
                 tp_type: type = float,
                 only_vertical_footprints: bool =False,
                 extensive_matching: bool = True,
                 save_path_ortho: str | None = None,
                 save_path_tps: str | None = None,
                 save_path_rot: str | None = None,
                 save_path_transform: str | None = None) -> (np.ndarray | None, list | None, int | None):
    """
    Georeference an orthophoto using tie points and transformations.

    This function processes an orthophoto created by Agisoft Metashape, trims and rotates the image, detects tie points
    between the orthophoto and satellite imagery, and calculates the transformation matrix for geo-referencing.

    Args:
        ortho (np.ndarray): The orthophoto to be georeferenced.
        image_ids (List[str]): List of image IDs corresponding to the footprints.
        footprints (List): List of footprint geometries for the images.
        lst_aligned (List[bool]): List indicating whether each image ID is aligned.
        azimuth (Optional[int]): Initial azimuth angle for rotation. Defaults to None.
        auto_rotate (bool): Whether to automatically determine the best rotation. Defaults to False.
        rotation_step (int): Step size for rotation in degrees. Defaults to 20.
        max_size (int): Maximum size for the orthophoto (in pixels). Defaults to 25000.
        trim_image (bool): Whether to trim the orthophoto to exclude white borders. Defaults to True.
        min_nr_tps (int): Minimum number of tie points required. Defaults to 100.
        tp_type (type): Data type for tie points. Defaults to float.
        only_vertical_footprints (bool): Whether to use only vertical footprints for geo-referencing. Defaults to False.
        save_path_ortho (Optional[str]): Path to save the transformed orthophoto. Defaults to None.
        save_path_tps (Optional[str]): Path to save the tie points. Defaults to None.
        save_path_rot (Optional[str]): Path to save the best rotation angle. Defaults to None.
        save_path_transform (Optional[str]): Path to save the transformation matrix. Defaults to None.

    Returns:
        Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]], Optional[int]]:
            - Transformation matrix as a 3x3 numpy array.
            - Geographic bounds after transformation as (min_x, min_y, max_x, max_y).
            - Best rotation angle in degrees.

    """

    # Ensure the orthophoto has 2 dimensions
    if len(ortho.shape) == 3:
        ortho = copy.deepcopy(ortho)
        ortho = ortho[0, :, :]

    # ensure confs are right
    if min_conf > start_conf:
        raise ValueError("min_conf must be smaller than start_conf")

    # init tie point detector
    tpd = ftp.TiePointDetector('lightglue',
                               min_conf=min_conf,
                               tp_type=tp_type,
                               verbose=True)

    # save the original size of the ortho
    original_height_ortho, original_width_ortho = ortho.shape[:2]

    # Trim the orthophoto if needed
    if trim_image:

        trim_threshold = .75

        # Calculate the fraction of white pixels for each row and column
        row_white_fraction = np.mean(ortho == 255, axis=1)
        col_white_fraction = np.mean(ortho == 255, axis=0)

        # Identify rows and columns that are not mostly white
        rows_to_keep = np.where(row_white_fraction < trim_threshold)[0]
        cols_to_keep = np.where(col_white_fraction < trim_threshold)[0]

        if rows_to_keep.size and cols_to_keep.size:
            # Determine the bounding box for rows and columns to keep
            trim_y_min = rows_to_keep[0]
            trim_y_max = rows_to_keep[-1] + 1  # +1 to include the last index
            trim_x_min = cols_to_keep[0]
            trim_x_max = cols_to_keep[-1] + 1

            # Trim the image using the calculated indices
            trimmed_ortho = ortho[trim_y_min:trim_y_max, trim_x_min:trim_x_max]

    else:
        trimmed_ortho = ortho
        trim_x_min = 0
        trim_y_min = 0

    # resize the ortho to a maximum size
    ortho_resized, s_x, s_y = rm.resize_max(trimmed_ortho, max_size=max_size)

    if sat_bounds is None:
        # Determine if there is at least one aligned vertical image
        if only_vertical_footprints:
            only_vertical_footprints = any("V" in image_id and aligned for image_id, aligned in zip(image_ids, lst_aligned))
        print(f"Only vertical footprints: {only_vertical_footprints}")

        # If so, update lst_aligned to keep only aligned vertical images
        if only_vertical_footprints:
            lst_aligned = ["V" in image_id and aligned for image_id, aligned in zip(image_ids, lst_aligned)]

        # Filter the footprints based on the (possibly updated) lst_aligned
        aligned_footprints = [fp for fp, aligned in zip(footprints, lst_aligned) if aligned]

        if len(aligned_footprints) == 0:
            raise ValueError("No aligned footprints found")

        if debug_save_footprints:
            print(aligned_footprints)
            pth = save_path_ortho.replace("ortho_georeferenced.tif", "footprints.geojson")
            eg.export_geometry(aligned_footprints, pth, overwrite_file=True)

        # Get the min and max for all footprints
        min_abs_x = min([footprint.bounds[0] for footprint in aligned_footprints])
        min_abs_y = min([footprint.bounds[1] for footprint in aligned_footprints])
        max_abs_x = max([footprint.bounds[2] for footprint in aligned_footprints])
        max_abs_y = max([footprint.bounds[3] for footprint in aligned_footprints])

        sat_bounds = (min_abs_x, min_abs_y, max_abs_x, max_abs_y)

    # load the satellite image
    sat, sat_transform = ls.load_satellite(sat_bounds, return_transform=True)

    if debug_save_sat and save_path_ortho is not None:
        # save the sat image
        pth = save_path_ortho.replace("ortho_georeferenced", "sat_for_georef")
        at.apply_transform(sat, sat_transform, save_path=pth)

    # Determine rotation values
    if azimuth is None and auto_rotate is False:
        # no azimuth but also no rotation (images are already aligned)
        rotation_values = [0]
    elif azimuth is None and auto_rotate is True:
        # rotate the ortho in steps of rotation_step
        rotation_values = list(range(0, 360, int(rotation_step)))
    elif azimuth is not None and auto_rotate is True:
        # azimuth is given, but we still want to rotate (for inaccurate azimuth)
        rotation_values = [
            azimuth - rotation_step,
            azimuth - rotation_step / 2,
            azimuth,
            azimuth + rotation_step / 2,
            azimuth + rotation_step
        ]
        # assure all entries are between 0 and 360
        rotation_values = [x % 360 for x in rotation_values]
    elif azimuth is not None and auto_rotate is False:
        rotation_values = [azimuth]
        # azimuth is given and we trust it
    else:
        raise ValueError("Invalid combination of azimuth and auto_rotate")

    # save the best value
    best_rot_mat = None
    best_rot = None
    best_tps = np.empty((0, 4))
    best_conf = np.empty((0, 1))

    #try different rotations
    for rotation in rotation_values:
        print(f"  - try rotation of {rotation}")

        # rotate the ortho
        ortho_rotated, rot_mat = ri.rotate_image(ortho_resized, rotation,
                                                 return_rot_matrix=True)
        # make no data uniform again
        ortho_rotated[ortho_rotated == 0] = 255

        # first find tie points globally
        tie_points, conf = tpd.find_tie_points(sat, ortho_rotated)

        # only required if not enough tie points are found
        if tie_points.shape[0] <= 100 and extensive_matching:

            # define patch dimensions and stride
            patch_h, patch_w = ortho_rotated.shape[0] // 2, ortho_rotated.shape[1] // 2
            stride_y, stride_x = patch_h // 2, patch_w // 2

            patch_counter = 0
            patch_tps = np.empty((0, 4))
            patch_conf = np.empty((0,))

            for y in range(3):
                for x in range(3):

                    patch_counter += 1

                    start_y, start_x = y * stride_y, x * stride_x
                    end_y, end_x = start_y + patch_h, start_x + patch_w

                    # get patch
                    ortho_patch = ortho_rotated[start_y:end_y, start_x:end_x]

                    # find tie points
                    tie_points_patch, conf_patch = tpd.find_tie_points(sat, ortho_patch)

                    if debug_show_patch_tie_points:
                        style_config = {
                            "title": f"{tie_points_patch.shape[0]} tie points "
                                     f"found at rotation {rotation} "
                                     f"for patch {patch_counter}",
                        }
                        di.display_images([sat, ortho_patch],
                                          tie_points=tie_points_patch, tie_points_conf=conf_patch,
                                          style_config=style_config)

                    # add offset to tie points
                    tie_points_patch[:, 2] += start_x
                    tie_points_patch[:, 3] += start_y

                    # add patch tie points to global tie points
                    if tie_points_patch.shape[0]> 0:
                        patch_tps = np.concatenate((patch_tps, tie_points_patch), axis=0)
                        patch_conf = np.concatenate((patch_conf, conf_patch), axis=0)

            if patch_tps.shape[0] > 1:
                # remove duplicates for column 1,2
                _, unique_indices = np.unique(patch_tps[:, :2], axis=0, return_index=True)
                patch_tps = patch_tps[np.sort(unique_indices)]
                patch_conf = patch_conf[np.sort(unique_indices)]

                # remove duplicates for column 3,4
                _, unique_indices = np.unique(patch_tps[:, 2:], axis=0, return_index=True)
                patch_tps = patch_tps[np.sort(unique_indices)]
                patch_conf = patch_conf[np.sort(unique_indices)]

            if patch_tps.shape[0] > tie_points.shape[0]:

                tie_points = patch_tps
                conf = patch_conf

        print("    ", tie_points.shape[0], "tie points found")

        # show the rotated tie points
        if debug_show_rotated_tie_points and tie_points.shape[0] > debug_min_tps:
            style_config = {
                "title": f"{tie_points.shape[0]} tie points "
                         f"found at rotation {rotation}",
            }
            di.display_images([sat, ortho_rotated],
                              tie_points=tie_points, tie_points_conf=conf,
                              style_config=style_config)

        # check if the new tie points are better than the old ones
        if tie_points.shape[0] > best_tps.shape[0]:
            best_tps = tie_points
            best_conf = conf
            best_rot_mat = rot_mat
            best_rot = rotation

    # get the best values
    tps = best_tps
    conf = best_conf
    rot_mat = best_rot_mat
    rotation = best_rot

    # that means no tie points were found
    if tps.shape[0] == 0:
        print(f"No tie points found")
        return None, sat_bounds, best_rot, 0

    # rotate points back
    tps[:, 2:] = rp.rotate_points(tps[:, 2:],
                                  rot_mat, invert=True)

    # account for the scaling
    tps[:, 2] = tps[:, 2] / s_x
    tps[:, 3] = tps[:, 3] / s_y

    # account for the trimming
    tps[:, 2] = tps[:, 2] + trim_x_min
    tps[:, 3] = tps[:, 3] + trim_y_min

    # show the final tie points
    if debug_show_final_tie_points:
        di.display_images([sat, ortho],
                          tie_points=tps, tie_points_conf=conf)

    # convert points to absolute values
    absolute_points = np.array([sat_transform * tuple(point) for point in tps[:, 0:2]])
    tps[:, 0:2] = absolute_points

    # set start confidence
    conf_val = start_conf

    # save all tps
    if save_path_tps is not None:

        pth = save_path_tps.replace(".csv", "_all.csv")
        column_names = "x_abs;y_abs;x_rel;y_rel;conf"
        np.savetxt(pth, tps, delimiter=';', header=column_names)

    # filter the tie points based on confidence with decreasing confidence
    while conf_val >= min_conf:

        # remove tie points with low confidence
        tps_filtered = tps[conf >= conf_val]

        print(f"  - {tps_filtered.shape[0]} tie points left with confidence {round(conf_val, 2)}")

        # check if we have enough points
        if tps_filtered.shape[0] >= min_nr_tps:
            break

        conf_val -= 0.05

    # use the filtered tie points
    tps = tps_filtered

    # save the tie points to a file
    if save_path_tps is not None:
        pth = save_path_tps.replace(".csv", "_filtered.csv")
        column_names = "x_abs;y_abs;x_rel;y_rel;conf"
        np.savetxt(pth, tps, delimiter=';', header=column_names)

    # check if we have enough points
    if tps.shape[0] < min_nr_tps:

        print(f"Not enough tie points ({tps.shape[0]}) found (min required: {min_nr_tps})")
        return None, sat_bounds, best_rot, tps.shape[0]

    # calculate the transform
    transform, residuals = ct.calc_transform(ortho, tps,
                                             transform_method="rasterio",
                                             gdal_order=3)

    if transform is None:
        return None, sat_bounds, best_rot, tps.shape[0]

    # apply the transform to the ortho
    if save_path_ortho is not None:
        # set the things we trim to 0
        ortho[ortho == 255] = 0

        # apply the transform
        at.apply_transform(ortho, transform, save_path_ortho)

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

    # get the transformed bounds
    transformed_bounds = (min_trans_x, min_trans_y, max_trans_x, max_trans_y)

    # save the best rotation
    if save_path_rot is not None:
        np.savetxt(save_path_rot, np.array([rotation]), delimiter=',')

    """
    print("Transform")
    print(transform)
    print("Transformed bounds")
    print(transformed_bounds)
    print("Rotation")
    print(rotation)
    print("ORTHO")
    print(ortho.shape)
    """

    # save the transform
    if save_path_transform is not None:
        np.savetxt(save_path_transform, transform, delimiter=',')

    return transform, transformed_bounds, rotation, tps.shape[0]
