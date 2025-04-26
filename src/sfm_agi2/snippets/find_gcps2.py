import copy
import numpy as np
import os
import pandas as pd

from rasterio.transform import Affine
from tqdm import tqdm

import src.base.find_tie_points as ftp
import src.base.resize_image as rei
import src.base.rotate_image as ri
import src.base.rotate_points as rp
import src.display.display_images as di

from src.sfm_agi2.SfMError import SfMError

def find_gcps(ortho_rel: np.ndarray, ortho_abs: np.ndarray,
              dem_rel: np.ndarray, dem_abs: np.ndarray,
              rotation_rel: float, transform_abs: Affine,
              resolution: float,
              mask_rel: np.ndarray | None = None,
              mask_rock: np.ndarray | None = None,
              slope: np.ndarray | None = None,
              resize_mode="none",
              min_gcps: int = 0,
              cell_size: int = 2000,
              min_conf: float = 0.9,
              extensive_search: bool = False,
              extensive_subset: int = 500,
              extensive_distance: int = 5,
              start_slope: int = 10,
              end_slope: int = 90,
              slope_step: int = 5,
              quadrant_size: int = 500, # in px
              tp_type=float,
              no_data_val: int = -9999,
              debug: bool = False,
              debug_folder: str | None = None,
              data_folder: str | None = None,
              ):

    # check shape of ortho abs
    if ortho_abs.ndim == 2:
        ortho_abs = ortho_abs[np.newaxis, :, :]
    elif ortho_abs.ndim == 3:
        if ortho_abs.shape[0] > 10:
            raise ValueError("Bands for the modern ortho should be in the "
                             "first dimension!")

    # assure that shapes of input data is correct
    if ortho_rel.shape != dem_rel.shape:
        raise ValueError("Relative ortho and DEM have different shapes")
            # check and create dummy masks

    # create dummy masks
    if mask_rel is None:
        mask_rel = np.ones(ortho_rel.shape, dtype=bool)
    if mask_rock is None:
        mask_rock = np.ones(ortho_abs.shape, dtype=bool)

    # create dummy slope
    if slope is None:
        slope = np.ones(ortho_abs.shape, dtype=bool)

    # create data folder if needed
    if data_folder is not None:
        os.makedirs(data_folder, exist_ok=True)

    # create debug folder if needed
    if debug and debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    # get 2d shape of ortho abs
    ortho_abs_shape = (ortho_abs.shape[1], ortho_abs.shape[2])

    # rotate relative data
    ortho_rel_rotated, rel_rot_mat = ri.rotate_image(ortho_rel, rotation_rel,
                                        return_rot_matrix=True)

    # resize the relative data to the absolute data
    if resize_mode == "rel":
        print("[INFO] Resize relative data to shape of absolute data")
        ortho_rel_resized, s_x_rel, s_y_rel = rei.resize_image(ortho_rel_rotated, ortho_abs_shape,
                                    return_scale_factor=True)
        ortho_abs_resized = ortho_abs
        s_x_abs, s_y_abs = 1, 1
    elif resize_mode == "abs":
        print("[INFO] Resize absolute data to shape of relative data")
        ortho_rel_resized = ortho_rel_rotated
        ortho_abs_resized, s_x_abs, s_y_abs = rei.resize_image(ortho_abs, ortho_rel_rotated.shape,
                                    return_scale_factor=True)
        s_x_rel, s_y_rel = 1, 1
    else:
        ortho_rel_resized = ortho_rel_rotated
        ortho_abs_resized = ortho_abs
        s_x_rel, s_y_rel = 1, 1
        s_x_abs, s_y_abs = 1, 1

    # init tie point detector
    tpd = ftp.TiePointDetector('lightglue',
                               min_conf=min_conf, tp_type=tp_type)

    # get the correct cell size
    if resize_mode in ["rel", "abs"]:
        n_x = int(np.ceil(ortho_rel_resized.shape[1] / cell_size))
        n_y = int(np.ceil(ortho_rel_resized.shape[0] / cell_size))
        cell_size_rel_x = cell_size
        cell_size_rel_y = cell_size
        cell_size_abs_x = cell_size
        cell_size_abs_y = cell_size
    else:
        # get the biggest shape
        nr_pixels_rel = ortho_rel_resized.shape[0] * ortho_rel_resized.shape[1]
        nr_pixels_abs = ortho_abs_resized.shape[1] * ortho_abs_resized.shape[2]

        # get nr of cells
        if nr_pixels_rel >= nr_pixels_abs:
            n_x = int(np.ceil(ortho_rel_resized.shape[1] / cell_size))
            n_y = int(np.ceil(ortho_rel_resized.shape[0] / cell_size))
            cell_size_rel_x = cell_size
            cell_size_rel_y = cell_size
            cell_size_abs_x = int(ortho_abs_resized.shape[2] / n_y)
            cell_size_abs_y = int(ortho_abs_resized.shape[1] / n_x)
        else:
            n_x = int(np.ceil(ortho_abs_resized.shape[2] / cell_size))
            n_y = int(np.ceil(ortho_abs_resized.shape[1] / cell_size))
            cell_size_rel_x = int(ortho_rel_resized.shape[0] / n_x)
            cell_size_rel_y = int(ortho_rel_resized.shape[1] / n_y)
            cell_size_abs_x = cell_size
            cell_size_abs_y = cell_size

    # init lists to store all tps
    lst_all_tps = []
    lst_all_conf = []

    # create progress bar
    pbar = tqdm(total=n_x * n_y, desc="Check cells for matches",
                position=0, leave=True)


    # iterate over all cells for the first search
    for i in range(n_x):
        for j in range(n_y):

            # update the progress bar description
            pbar.set_postfix_str(f"Cell: {i}, {j}")

            # get cell size for rel
            x_min_rel = i * cell_size_rel_x
            x_max_rel = min((i + 1) * cell_size_rel_x,
                            ortho_rel_resized.shape[1])
            y_min_rel = j * cell_size_rel_y
            y_max_rel = min((j + 1) * cell_size_rel_y,
                            ortho_rel_resized.shape[0])

            # get cell bounds for abs
            x_min_abs = i * cell_size_abs_x
            x_max_abs = min((i + 1) * cell_size_abs_x,
                            ortho_abs_resized.shape[2])
            y_min_abs = j * cell_size_abs_x
            y_max_abs = min((j + 1) * cell_size_abs_y,
                            ortho_abs_resized.shape[1])

            # get cell data
            abs_cell = ortho_abs_resized[:, y_min_abs:y_max_abs, x_min_abs:x_max_abs]
            rel_cell = ortho_rel_resized[y_min_rel:y_max_rel, x_min_rel:x_max_rel]

            # make sure the cells are correct (no cells with 0 size)
            if abs_cell.shape[0] == 0 or abs_cell.shape[1] == 0 or \
                    rel_cell.shape[0] == 0 or rel_cell.shape[1] == 0:
                raise ValueError("Cell is empty, check cell size and image size")

            # skip if completely empty
            if np.all(rel_cell == 255):
                lst_all_tps.append(np.empty((0, 4)))
                lst_all_conf.append(np.empty((0, 1)))
                pbar.update(1)
                continue

            # find tie points between the two cells
            tps, conf = tpd.find_tie_points(abs_cell, rel_cell)

            # save debug images
            if debug:
                dbg_file_name = f"cell_{i}_{j}.png"
                dbg_img_path = os.path.join(debug_folder, dbg_file_name)
                style_config = {
                    "title": f"{tps.shape[0]} tps for cell {i}, {j}",
                }

                di.display_images([abs_cell, rel_cell],
                                  tie_points=tps,
                                  tie_points_conf=conf,
                                  style_config=style_config,
                                  save_path=dbg_img_path)

            # skip if no tps are found in the cell
            if tps.shape[0] == 0:
                # append also empty values
                lst_all_tps.append(np.empty((0, 4)))
                lst_all_conf.append(np.empty((0, 1)))
                pbar.update(1)
                continue

            # add the cell offset
            tps[:, 0] += x_min_abs
            tps[:, 1] += y_min_abs
            tps[:, 2] += x_min_rel
            tps[:, 3] += y_min_rel

            # append to lists for all tps / conf
            lst_all_tps.append(tps)
            lst_all_conf.append(conf)

            # update progress bar
            pbar.update(1)

    # close pbar
    pbar.set_postfix_str("- Finished -")
    pbar.close()

    # convert list of arrays to array
    tps = np.vstack(lst_all_tps)
    conf = np.vstack([a.reshape(-1, 1) for a in lst_all_conf])

    # raise an error if no tie points are found at all
    if tps.shape[0] == 0:
        raise SfMError("No tie points found when looking for GCPs")

    print(f"[INFO] Found {tps.shape[0]} tie points in the search")

    # do an extensive search as well
    extensive_search = True
    if extensive_search:
        pbar = tqdm(total=tps.shape[0], desc="Extensive search for matches",
                    position=0, leave=True)

        lst_extensive_tps = []
        lst_extensive_conf = []

        for i in range(tps.shape[0]):

            # get current tp
            tp = tps[i]

            # get the absolute subset for that tp
            x_min_abs = max(int(tp[0] - extensive_subset / 2), 0)
            x_max_abs = min(int(tp[0] + extensive_subset / 2),
                            ortho_abs_resized.shape[2])
            y_min_abs = max(int(tp[1] - extensive_subset / 2), 0)
            y_max_abs = min(int(tp[1] + extensive_subset / 2),
                            ortho_abs_resized.shape[1])
            abs_subset = ortho_abs_resized[:,
                         y_min_abs:y_max_abs, x_min_abs:x_max_abs]

            # get the relative subset for that tp
            x_min_rel = max(int(tp[2] - extensive_subset / 2), 0)
            x_max_rel = min(int(tp[2] + extensive_subset / 2),
                            ortho_rel_resized.shape[1])
            y_min_rel = max(int(tp[3] - extensive_subset / 2), 0)
            y_max_rel = min(int(tp[3] + extensive_subset / 2),
                            ortho_rel_resized.shape[0])
            rel_subset = ortho_rel_resized[y_min_rel:y_max_rel,
                         x_min_rel:x_max_rel]

            # find matches for the subset
            tps_subset, conf_subset = tpd.find_tie_points(abs_subset,
                                                          rel_subset)

            if tps_subset.shape[0] == 0:
                pbar.update(1)
                continue

            # add the cell offset
            tps_subset[:, 0] += x_min_abs
            tps_subset[:, 1] += y_min_abs
            tps_subset[:, 2] += x_min_rel
            tps_subset[:, 3] += y_min_rel

            # append to lists for all tps / conf
            lst_extensive_tps.append(tps_subset)
            lst_extensive_conf.append(conf_subset)

            # update progress bar
            pbar.update(1)

        # close pbar
        pbar.set_postfix_str("- Finished -")
        pbar.close()

        # merge the list
        if len(lst_extensive_tps) > 0:
            ext_tps = np.vstack(lst_extensive_tps)
            ext_conf = np.vstack([a.reshape(-1, 1) for a in lst_extensive_conf])
        else:
            # raise error, because when having normal tps, extensive should give even more
            raise SfMError("No tie points found in extensive search")

        # merge the two tp and conf arrays
        tps = np.vstack((tps, ext_tps))
        conf = np.vstack((conf, ext_conf))

        # sort by confidence descending
        order = np.argsort(conf[:, 0])[::-1]
        tps_ordered = tps[order]
        conf_ordered = conf[order]

        # deduplicate based on proximity in abs coordinates
        selected_tps = []
        selected_conf = []
        selected_mask = np.zeros(ortho_abs_shape, dtype=bool)

        # remove duplicates in the tps
        for tp, conf in zip(tps_ordered, conf_ordered):
            x, y = int(tp[0]), int(tp[1])

            x_min = max(0, x - extensive_distance)
            x_max = min(selected_mask.shape[1], x + extensive_distance + 1)
            y_min = max(0, y - extensive_distance)
            y_max = min(selected_mask.shape[0], y + extensive_distance + 1)

            if np.any(selected_mask[y_min:y_max, x_min:x_max]):
                continue  # too close to existing TP

            # mark selection
            selected_mask[y, x] = True
            selected_tps.append(tp)
            selected_conf.append(conf)

        # convert to arrays
        tps = np.vstack(selected_tps)
        conf = np.vstack(selected_conf)

        if tps.shape[0] == 0:
            raise SfMError("No tie points found in extensive search")

        print(f"[INFO] Found {tps.shape[0]} tie points in extensive search")

    # rescale the tie points to the original image size
    tps[:, 0] = tps[:, 0] / s_x_abs
    tps[:, 1] = tps[:, 1] / s_y_abs
    tps[:, 2] = tps[:, 2] / s_x_rel
    tps[:, 3] = tps[:, 3] / s_y_rel

    # rotate the tie points back
    tps[:, 2:] = rp.rotate_points(tps[:, 2:], rel_rot_mat, invert=True)

    # filter by masks
    rock_coords_x, rock_coords_y = _get_new_coords(tps[:, 0], tps[:, 1],
                                ortho_abs_shape, mask_rock.shape)
    rock_vals = mask_rock[rock_coords_y, rock_coords_x]
    rel_vals = mask_rel[np.round(tps[:, 3]).astype(int),
                        np.round(tps[:, 2]).astype(int)]
    masked_indices = rel_vals & rock_vals

    debug_display_masked_tps = False
    if debug_display_masked_tps:
        rock_coords = np.column_stack([rock_coords_x, rock_coords_y])
        di.display_images([ortho_abs, ortho_rel, mask_rock, mask_rel],
                          points=[tps[:,:2], tps[:,2:4], rock_coords, tps[:, 2:4]])

    # filter tps and conf
    tps = tps[masked_indices]
    conf = conf[masked_indices]


    if tps.shape[0] < min_gcps:
        raise SfMError(f"Too few tie points already after filtering by mask ({tps.shape[0]} < {min_gcps})")

    slope_coords_x, slope_coords_y = _get_new_coords(tps[:, 0], tps[:, 1],
                                   ortho_abs_shape, slope.shape)
    slope_vals = slope[slope_coords_y, slope_coords_x]

    # divide into quadrants
    H, W = ortho_rel.shape
    quadrant_divider_h = max(1, H // quadrant_size)
    quadrant_divider_w = max(1, W // quadrant_size)
    q_h = H // quadrant_divider_h
    q_w = W // quadrant_divider_w

    quadrant_tps = []
    quadrant_conf = []
    quadrant_slope = []

    for qy in range(quadrant_divider_h):
        for qx in range(quadrant_divider_w):

            # Quadrant bounds
            x_min_q = qx * q_w
            x_max_q = (qx + 1) * q_w if qx < quadrant_divider_w - 1 else W
            y_min_q = qy * q_h
            y_max_q = (qy + 1) * q_h if qy < quadrant_divider_h - 1 else H

            # Create mask for points within this quadrant (based on x_rel, y_rel)
            in_quadrant_mask = (
                    (tps[:, 2] >= x_min_q) & (tps[:, 2] < x_max_q) &
                    (tps[:, 3] >= y_min_q) & (tps[:, 3] < y_max_q)
            )
            # Apply mask to get the actual points
            tps_q = tps[in_quadrant_mask]
            conf_q = conf[in_quadrant_mask]
            slope_q = slope_vals[in_quadrant_mask]

            if tps_q.shape[0] == 0:
                continue

            # get at least one point in each quadrant possible
            current_slope = copy.deepcopy(start_slope)
            while current_slope <= end_slope:

                # get valid and invalid indices
                valid_indices = np.where(slope_q < current_slope)[0]

                # add valid indices
                if len(valid_indices) > 0:
                    quadrant_tps.append(tps_q[valid_indices])
                    quadrant_conf.append(conf_q[valid_indices])
                    quadrant_slope.append(slope_q[valid_indices])
                    break

                # set the slope value
                current_slope += slope_step

    # Flatten and check final number of GCPs
    tps_selected = np.vstack(quadrant_tps) if quadrant_tps else np.empty((0, 4))
    conf_selected = np.vstack(quadrant_conf) if quadrant_conf else np.empty((0, 1))
    slope_selected = np.hstack(quadrant_slope) if quadrant_slope else np.empty((0,))

    # If too few, select more from original tps based on lowest slope
    if tps_selected.shape[0] < min_gcps:
        print(f"[INFO] Only {tps_selected.shape[0]} GCPs found, adding more points with lowest slope...")

        # Create a mask for already selected points
        selected_mask = np.zeros(tps.shape[0], dtype=bool)
        for i in range(tps_selected.shape[0]):
            match = np.all(np.isclose(tps, tps_selected[i], atol=1e-3), axis=1)
            selected_mask = selected_mask | match

        # Remaining unselected points
        unselected_indices = np.where(~selected_mask)[0]
        if len(unselected_indices) > 0:
            unselected_slope = slope_vals[unselected_indices]
            sorted_idx = unselected_indices[np.argsort(unselected_slope)]

            needed = min_gcps - tps_selected.shape[0]
            add_indices = sorted_idx[:needed]

            tps_selected = np.vstack([tps_selected, tps[add_indices]])
            conf_selected = np.vstack([conf_selected, conf[add_indices]])
            slope_selected = np.hstack([slope_selected, slope_vals[add_indices]])

    if tps_selected.shape[0] < min_gcps:
        raise SfMError(f"Too few tie points left after filtering and fallback: {tps_selected.shape[0]} < {min_gcps}")

    # Final result
    tps = tps_selected
    conf = conf_selected
    slope_vals = slope_selected

    # get px coords for dems
    dem_new_coords_x, dem_new_coords_y = _get_new_coords(tps[:, 0], tps[:, 1],
                                     ortho_abs_shape, dem_abs.shape)
    px_coords = np.round(tps[:, 2:4]).astype(int)

    # make relative and absolute coordinates
    relative_coords = tps[:, 2:4] * resolution
    absolute_coords = np.array([transform_abs * tuple(point) for point in tps[:, 0:2]])

    # get elevation values
    elevations_rel = dem_rel[px_coords[:, 1], px_coords[:, 0]]
    elevations_abs = dem_abs[dem_new_coords_y, dem_new_coords_x]

    debug_display_elevations = False
    if debug_display_elevations:
        abs_coords = np.column_stack([dem_new_coords_x, dem_new_coords_y])
        rel_coords = np.column_stack([px_coords[:, 0], px_coords[:, 1]])
        di.display_images([dem_abs, dem_rel],
                          image_types=["dem", "dem"],
                          points=[abs_coords, rel_coords])


    # stack to new array
    stacked_arr = np.hstack([
        absolute_coords,                         # (2, 2)
        elevations_abs.reshape(-1, 1),           # (2, 1)
        relative_coords,                         # (2, 2)
        elevations_rel.reshape(-1, 1),           # (2, 1)
        px_coords,                               # (2, 2)
        conf.reshape(-1, 1),                     # (2, 1)
        slope_vals.reshape(-1, 1)                # (2, 1)
    ])

    print(stacked_arr)

    # create a dataframe with the gcps
    df = pd.DataFrame(data=stacked_arr,
                      columns=["x_abs", "y_abs", "z_abs",
                               "x_rel", "y_rel", "z_rel",
                               "x_px", "y_px", "conf", "slope"])

    # remove entries with invalid z-values
    df = df[df["z_rel"] != no_data_val]

    # create a column with incremental gcp name
    gcp_names = [f"GCP_{i}" for i in range(1, df.shape[0] + 1)]
    df.insert(0, "GCP", gcp_names)

    # save gcps as csv
    if data_folder is not None:
        df.to_csv(os.path.join(data_folder, "gcps.csv"),
                  index=False)

    print(df)

    return df


def _get_new_coords(x: np.ndarray, y: np.ndarray,
                                  original_shape: tuple[int, int],
                                  target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Rescales coordinates (x, y) from `original_shape` to `target_shape`.

    Parameters:
    - x, y: arrays of coordinates in the original image space.
    - original_shape: (height, width) of the source array (e.g., ortho_abs_resized).
    - target_shape: (height, width) of the target array (e.g., mask_rock).

    Returns:
    - x_new, y_new: transformed coordinates in the target image space.
    """
    H_old, W_old = original_shape
    H_new, W_new = target_shape

    scale_x = W_new / W_old
    scale_y = H_new / H_old

    x_new = x * scale_x
    y_new = y * scale_y

    x_new = np.round(x_new)
    y_new = np.round(y_new)

    return x_new.astype(int), y_new.astype(int)
