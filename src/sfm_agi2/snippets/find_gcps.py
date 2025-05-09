import numpy as np
import os
import pandas as pd
from rasterio.transform import Affine
from scipy.ndimage import map_coordinates
from tqdm import tqdm

import src.base.find_tie_points as ftp
import src.base.resize_image as rei
import src.base.rotate_image as ri
import src.base.rotate_points as rp
import src.display.display_images as di
from src.sfm_agi2.SfMError import SfMError

debug_show_masks = False

def find_gcps(ortho_rel: np.ndarray,
              ortho_abs: np.ndarray,
              dem_rel: np.ndarray,
              dem_abs: np.ndarray,
              rotation_rel: float,
              transform_abs : Affine,
              resolution: float,
              mask_rel: np.ndarray | None = None,
              mask_rock: np.ndarray | None = None,
              slope: np.ndarray | None = None,
              resize_mode="none",
              cell_size: int = 2000,
              min_conf: float = 0.9,
              extensive_search: bool = False,
              extensive_subset: int = 500,
              extensive_distance: int = 5,
              selection_radius:int = 150,
              tp_type=float,
              existing_tps: np.ndarray | None = None,
              existing_conf: np.ndarray | None = None,
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

    # create data folder if needed
    if data_folder is not None:
        os.makedirs(data_folder, exist_ok=True)

    # create debug folder if needed
    if debug and debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    # check and create dummy masks
    if mask_rel is not None:
        if mask_rel.shape != dem_rel.shape:
            raise ValueError("Relative Mask and DEM have different shapes")
    else:
        mask_rel = np.ones(ortho_rel.shape, dtype=bool)
    if mask_rock is not None:
        if mask_rock.shape != dem_abs.shape:
            raise ValueError("Rock mask and DEM have different shapes")
    else:
        mask_rock = np.ones(ortho_abs.shape, dtype=bool)
    if slope is not None:
        if slope.shape != dem_abs.shape:
            raise ValueError("Slope and DEM have different shapes")
    else:
        slope = np.zeros(ortho_abs.shape, dtype=bool)

    # get 2d shape of ortho abs
    ortho_abs_shape = (ortho_abs.shape[1], ortho_abs.shape[2])

    # resize absolute mask to shape of ortho
    mask_abs = rei.resize_image(mask_abs, ortho_abs_shape)
    slope = rei.resize_image(slope, ortho_abs_shape)

    # rotate the relative data
    ortho_rel_rotated, rel_rot_mat = ri.rotate_image(ortho_rel, rotation_rel,
                                        return_rot_matrix=True)
    mask_rel_rotated = ri.rotate_image(mask_rel, rotation_rel)

    # update rotated mask
    mask_rel_rotated[ortho_rel_rotated == 255] = 0

    # resize the relative data to the absolute data
    if resize_mode == "rel":
        print("[INFO] Resize relative data to shape of absolute data")
        ortho_rel_resized, s_x_rel, s_y_rel = rei.resize_image(ortho_rel_rotated, ortho_abs_shape,
                                    return_scale_factor=True)
        mask_rel_resized= rei.resize_image(mask_rel_rotated, ortho_abs_shape)
        ortho_abs_resized = ortho_abs
        mask_abs_resized = mask_abs
        s_x_abs, s_y_abs = 1, 1
    elif resize_mode == "abs":
        print("[INFO] Resize absolute data to shape of relative data")
        ortho_rel_resized = ortho_rel_rotated
        mask_rel_resized = mask_rel_rotated
        ortho_abs_resized, s_x_abs, s_y_abs = rei.resize_image(ortho_abs, ortho_rel_rotated.shape,
                                    return_scale_factor=True)
        mask_abs_resized = rei.resize_image(mask_abs, ortho_rel_rotated.shape)
        s_x_rel, s_y_rel = 1, 1
    else:
        ortho_rel_resized = ortho_rel_rotated
        mask_rel_resized = mask_rel_rotated
        ortho_abs_resized = ortho_abs
        mask_abs_resized = mask_abs
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

    # check if we have already tie points
    if existing_tps is None:

        # init lists to stor all and filtered values
        lst_all_tps = []
        lst_all_conf = []

        # create progress bar
        pbar = tqdm(total=n_x*n_y, desc="Check cells for matches",
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

                # get mask cell data
                mask_abs_cell = mask_abs_resized[y_min_abs:y_max_abs, x_min_abs:x_max_abs]
                mask_rel_cell = mask_rel_resized[y_min_rel:y_max_rel, x_min_rel:x_max_rel]

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

                # skip if completely masked
                if np.all(mask_rel_cell == 0) or np.all(mask_abs_cell == 0):
                    lst_all_tps.append(np.empty((0, 4)))
                    lst_all_conf.append(np.empty((0, 1)))
                    pbar.update(1)
                    continue

                # find tie points
                if existing_tps is None:
                    tps, conf = tpd.find_tie_points(abs_cell, rel_cell)
                else:
                    tps = existing_tps[i * n_y + j]
                    conf = existing_conf[i * n_y + j]

                if debug:
                    dbg_file_name = f"cell_{i}_{j}.png"
                    dbg_img_path = os.path.join(debug_folder, dbg_file_name)
                    style_config={
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

        print(lst_all_conf)

        # convert list of arrays to array
        orig_tps = np.vstack(lst_all_tps)
        orig_conf = np.vstack([a.reshape(-1, 1) for a in lst_all_conf])

        # raise an error if no tie points are found at all
        if orig_tps.shape[0] == 0:
            raise SfMError("No tie points found when looking for GCPs")

        print(f"[INFO] Found {orig_tps.shape[0]} tie points in the search")

        # do an extensive search as well
        if extensive_search:
            pbar = tqdm(total=orig_tps.shape[0], desc="Extensive search for matches",
                        position=0, leave=True)

            lst_extensive_tps = []
            lst_extensive_conf = []

            for i in range(orig_tps.shape[0]):

                # get current tp
                tp = orig_tps[i]

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
                all_ext_tps = np.vstack(lst_extensive_tps)
                all_ext_conf = np.concatenate(lst_extensive_conf)
            else:
                # raise error, because when having normal tps, extensive should give even more
                raise SfMError("No tie points found in extensive search")

            # sort by confidence descending
            order = np.argsort(all_ext_conf)[::-1]
            all_ext_tps = all_ext_tps[order]
            all_ext_conf = all_ext_conf[order]

            # deduplicate based on proximity in abs coordinates
            selected_tps = []
            selected_conf = []
            selected_mask = np.zeros((ortho_abs.shape[1], ortho_abs.shape[2]), dtype=bool)

            for tp, conf in zip(all_ext_tps, all_ext_conf):
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
            orig_tps = np.vstack(selected_tps)
            orig_conf = np.vstack(selected_conf)

            if orig_tps.shape[0] == 0:
                raise SfMError("No tie points found in extensive search")

            print(f"[INFO] Found {orig_tps.shape[0]} tie points in extensive search")
    else:
        orig_tps = existing_tps
        orig_conf = existing_conf

    # convert to integers
    x_abs = orig_tps[:, 0].astype(int)
    y_abs = orig_tps[:, 1].astype(int)
    x_rel = orig_tps[:, 2].astype(int)
    y_rel = orig_tps[:, 3].astype(int)

    # apply mask
    mask_ok = (
        (x_abs >= 0) & (x_abs < mask_abs_resized.shape[1]) &
        (y_abs >= 0) & (y_abs < mask_abs_resized.shape[0]) &
        (x_rel >= 0) & (x_rel < mask_rel_resized.shape[1]) &
        (y_rel >= 0) & (y_rel < mask_rel_resized.shape[0]) &
        mask_abs_resized[y_abs, x_abs] &
        mask_rel_resized[y_rel, x_rel]
    )
    tps_masked = orig_tps[mask_ok]
    conf_masked = orig_conf[mask_ok]

    if debug_show_masks:
        di.display_images([slope, mask_abs_resized, mask_rel_resized],
                          points=[orig_tps[:, 0:2], orig_tps[:, 0:2], orig_tps[:, 2:4]])

    # we can end here if all tps are masked
    if tps_masked.shape[0] == 0:
        return np.empty((0, 4)), orig_tps, orig_conf

    # reset naming
    tps = tps_masked
    conf = conf_masked

    # rescale the tie-points back to their original resolution
    tps[:, 0] /= s_x_abs
    tps[:, 1] /= s_y_abs
    tps[:, 2] /= s_x_rel
    tps[:, 3] /= s_y_rel

    # rotate the tie points back
    tps[:, 2:] = rp.rotate_points(tps[:, 2:], rel_rot_mat, invert=True)

    # order the tie points by confidence
    order = np.argsort(conf.ravel())[::-1]
    tps = tps[order]

    # Precompute a circular mask for the selection radius
    y_grid, x_grid = np.ogrid[-selection_radius:selection_radius + 1,
                     -selection_radius:selection_radius + 1]
    circular_mask = x_grid ** 2 + y_grid ** 2 <= selection_radius ** 2

    # variables for tp selection
    filtered_tps, filtered_conf = [], []
    selection_mask = np.zeros(ortho_abs_shape, dtype=bool)

    # iterate over tie-points
    pbar = tqdm(total=tps.shape[0], desc="Filter GCPs",
                position=0, leave=True)

    # filter the tps
    for i in range(tps.shape[0]):

        # get x,y of the tie point
        x, y = int(tps[i, 0]), int(tps[i, 1])

        # update the progress bar description
        pbar.set_postfix_str("GCP {} at ({},{})".format(i, x, y))

        # Check if the tie point falls within an already selected region
        if selection_mask[y, x] == 1:
            pbar.update(1)
            continue

        filtered_tps.append(tps[i])
        filtered_conf.append(conf[i])

        # define bounds
        y_min = max(0, y - selection_radius)
        y_max = min(selection_mask.shape[0], y + selection_radius + 1)
        x_min = max(0, x - selection_radius)
        x_max = min(selection_mask.shape[1], x + selection_radius + 1)

        # extract the corresponding region in the circular mask
        cm_y_start = max(0, selection_radius - y)
        cm_y_end = cm_y_start + (y_max - y_min)
        cm_x_start = max(0, selection_radius - x)
        cm_x_end = cm_x_start + (x_max - x_min)
        circular_region = circular_mask[cm_y_start:cm_y_end, cm_x_start:cm_x_end]

        # apply the mask
        selection_mask[y_min:y_max, x_min:x_max] |= circular_region

    # if filtering is too restrictive just return the original tps
    if len(filtered_tps) == 0:
        return np.empty((0, 4)), orig_tps, orig_conf

    # create a new array with the filtered tie points
    tps = np.vstack(filtered_tps)
    conf = np.vstack(filtered_conf)

    # get the pixel coordinates
    px_coords = tps[:, 2:4]
    relative_coords = px_coords * resolution

    # Normalize ortho_abs pixel coordinates
    px_norm_x = tps[:, 0] / ortho_abs.shape[2]
    px_norm_y = tps[:, 1] / ortho_abs.shape[1]

    # Scale to dem_abs coordinates
    x_dem = px_norm_x * dem_abs.shape[1]
    y_dem = px_norm_y * dem_abs.shape[0]

    # get abs elevations
    coords = np.vstack((y_dem, x_dem))
    elevations_abs = map_coordinates(dem_abs, coords, order=1,
                                     mode="nearest").reshape(-1, 1)

    # get the old elevation
    coords = np.vstack((tps[:, 3], tps[:, 2]))
    elevations_rel = map_coordinates(dem_rel, coords, order=1,
                                     mode="nearest").reshape(-1, 1)

    # get absolute coords for the tie points
    absolute_coords = np.array([transform_abs * tuple(point) for point in tps[:, 0:2]])
    tps[:, 0:2] = absolute_coords

    # stack to new array
    stacked_arr = np.hstack([absolute_coords, elevations_abs,
                             relative_coords, elevations_rel,
                             px_coords, conf])

    # create a dataframe with the gcps
    df = pd.DataFrame(data=stacked_arr,
                      columns=["x_abs", "y_abs", "z_abs",
                               "x_rel", "y_rel", "z_rel",
                               "x_px", "y_px", "conf"])

    # remove entries with invalid z-values (TODO: REPLACE WITH NO DATA VAL)
    df = df[df["z_rel"] != -9999]

    # create a column with incremental gcp name
    gcp_names = [f"GCP_{i}" for i in range(1, df.shape[0] + 1)]
    df.insert(0, "GCP", gcp_names)

    # save gcps as csv
    if data_folder is not None:
        df.to_csv(os.path.join(data_folder, "gcps.csv"),
                  index=False)

    return df, orig_tps, orig_conf