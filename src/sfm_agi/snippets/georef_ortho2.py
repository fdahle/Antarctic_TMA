import copy

import numpy as np

import src.base.find_tie_points as ftp
import src.base.resize_image as rei
import src.base.rotate_image as ri
import src.base.rotate_points as rp
import src.load.load_satellite as ls
import src.georef.snippets.apply_transform as at
import src.georef.snippets.calc_transform as ct


def georef_ortho_2(ortho, transform, best_rot,
                   min_nr_tps=100, tp_type=float,
                   min_conf=0.75,
                   trim_threshold=0.75,
                   save_path_tps=None, save_path_ortho=None,
                   save_path_transform=None):

        print("Georef Ortho 2")

        # get the corners of the ortho image
        corners = np.array([[0, 0],
                            [0, ortho.shape[0]],
                            [ortho.shape[1], ortho.shape[0]],
                            [ortho.shape[1], 0]])

        # apply the transform to the corners
        transformed_corners = np.array([transform * (x, y) for x, y in corners])

        # get the bounds of the ortho image
        min_x = transformed_corners[:, 0].min()
        max_x = transformed_corners[:, 0].max()
        min_y = transformed_corners[:, 1].min()
        max_y = transformed_corners[:, 1].max()
        bounds = (min_x, min_y, max_x, max_y)

        # load satellite image with calculated new bounds
        sat, sat_transform = ls.load_satellite(bounds, return_transform=True)

        # copy original ortho
        orig_ortho = ortho.copy()

        # resize the ortho image to the size of the satellite image
        ortho, scale_x, scale_y = rei.resize_image(ortho, (sat.shape[1], sat.shape[2]), return_scale_factor=True)

        # rotate the ortho image
        ortho, rot_mat = ri.rotate_image(ortho, best_rot, return_rot_matrix=True)

        # Calculate the fraction of white pixels for each row and column
        row_white_fraction = np.mean(ortho == 0, axis=1)
        col_white_fraction = np.mean(ortho == 0, axis=0)

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
            trimmed_sat = sat[:, trim_y_min:trim_y_max, trim_x_min:trim_x_max]
        else:
            trimmed_ortho = ortho
            trimmed_sat = sat
            trim_x_min = 0
            trim_y_min = 0

        tpd = ftp.TiePointDetector('lightglue',
                                   min_conf=min_conf,
                                   tp_type=tp_type,
                                   verbose=True)
        tps, conf = tpd.find_tie_points(trimmed_sat, trimmed_ortho)

        # check if we have enough points
        if tps.shape[0] < min_nr_tps:
            print(f"Not enough tie points ({tps.shape[0]}) found in go2")
            return None, None, None

        # account for the trimming
        tps[:, 0] += trim_x_min
        tps[:, 1] += trim_y_min
        tps[:, 2] += trim_x_min
        tps[:, 3] += trim_y_min

        # rotate points back
        tps[:, 2:] = rp.rotate_points(tps[:, 2:],
                                      rot_mat, invert=True)

        # account for the resizing
        tps[:, 2] /= scale_x
        tps[:, 3] /= scale_y

        # get absolute points for the satellite image
        absolute_points = np.array([sat_transform * tuple(point) for point in tps[:, 0:2]])
        tps[:, 0:2] = absolute_points

        print(f"Found {tps.shape[0]} tie points")

        # save the tie points
        if save_path_tps is not None:
            np.savetxt(save_path_tps, tps, delimiter=";")

        # calculate the transform
        transform, _ = ct.calc_transform(orig_ortho, tps,
                                                     transform_method="rasterio",
                                                     gdal_order=3)

        # apply the transform to the ortho
        if save_path_ortho is not None:
            # set the things we trim to 0
            ortho[ortho == 255] = 0

            # apply the transform
            at.apply_transform(orig_ortho, transform, save_path_ortho)

        print("Transform")
        print(transform)
        print("ORTHO")
        print(ortho.shape)

        # save the transform
        if save_path_transform is not None:
            np.savetxt(save_path_transform, transform, delimiter=',')

        return transform