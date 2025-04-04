import numpy as np

import src.base.find_tie_points as ftp

def find_gcps(ortho_rel, ortho_new,
              dem_old, dem_new,
              mask_old=None, mask_new=None,
              min_conf=0.9,
              cell_size=2000,
              tp_type=float):

    # assure that shapes of input data is correct
    if ortho_rel.shape != dem_old.shape:
        raise ValueError("Relative ortho and old DEM have different shapes")
    if ortho_new.shape != dem_new.shape:
        raise ValueError("Absolute ortho and new DEM have different shapes")

    # check and create dummy mask
    if mask_old is not None:
        if mask_old.shape != dem_old.shape:
            raise ValueError("Mask and old DEM have different shapes")
    else:
        mask_old = np.ones(ortho_rel.shape, dtype=bool)
    if mask_new is not None:
        if mask_new.shape != dem_new.shape:
            raise ValueError("Mask and new DEM have different shapes")
    else:
        mask_new = np.ones(ortho_new.shape, dtype=bool)

    # init tie point detector
    tpd = ftp.TiePointDetector('lightglue',
                               min_conf=min_conf, tp_type=tp_type)

    # get nr of cells
    n_x = int(np.ceil(ortho_rel.shape[1] / cell_size))
    n_y = int(np.ceil(ortho_rel.shape[0] / cell_size))

    all_tps = []
    all_conf = []

    # iterate over all cells
    for i in range(n_x):
        for j in range(n_y):

            # get cell bounds
            x_min = i * cell_size
            x_max = min((i + 1) * cell_size, ortho_rel.shape[1])
            y_min = j * cell_size
            y_max = min((j + 1) * cell_size, ortho_rel.shape[0])

            # get cell data
            rel_cell = ortho_rel[y_min:y_max, x_min:x_max]
            abs_cell = ortho_abs[y_min:y_max, x_min:x_max]

            mask_old_cell = mask_old[y_min:y_max, x_min:x_max]
            mask_new_cell = mask_new[y_min:y_max, x_min:x_max]

            # skip if completely masked
            if np.all(mask_old_cell == 0) or np.all(mask_new_cell == 0):
                continue

            # find tie points
            tps, conf = tpd.find_tie_points(rel_cell, abs_cell)

            # apply mask
            TODO

            # add the cell offset
            tps[:, 0] += x_min
            tps[:, 1] += y_min
            tps[:, 2] += x_min
            tps[:, 3] += y_min

            # append to all lists
            all_tps.append(tps)
            all_conf.append(conf)

    if len(all_tps) > 0:
        tps = np.concatenate(all_tps)
        conf = np.concatenate(all_conf)
    else:
        raise ValueError("No tie points found when looking for GCPs")


    # order the tie points by confidence
    order = np.argsort(conf)[::-1]
    tps = tps[order]

    # Precompute a circular mask for the selection radius
    y_grid, x_grid = np.ogrid[-selection_radius:selection_radius + 1, -selection_radius:selection_radius + 1]
    circular_mask = x_grid ** 2 + y_grid ** 2 <= selection_radius ** 2


    if tps.shape[0] < min_gcps:
        raise ValueError("Not enough GCPs ({}/{})found".format(
            tps.shape[0], min_gcps))

    # get the modern elevations
    x_coords_new = tps[:, 0].astype(int)
    y_coords_new = tps[:, 1].astype(int)
    elevations_new = dem_new[y_coords_new, x_coords_new]
    elevations_new = elevations_new.reshape(-1, 1)

    # get the old elevation
    x_coords_old = tps[:, 2].astype(int)
    y_coords_old = tps[:, 3].astype(int)
    elevations_old = dem_old[y_coords_old, x_coords_old]
    elevations_old = elevations_old.reshape(-1, 1)

    # get absolute coords for the tie points
    ones = np.ones((tps.shape[0], 1))
    abs_px = np.hstack([tps[:, 0:2], ones])
    absolute_coords = abs_px @ transform.T[:, :2]

    return df