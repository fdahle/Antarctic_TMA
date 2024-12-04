import numpy as np
import pandas as pd

import src.base.find_tie_points as ftp
import src.base.resize_image as ri
import src.base.resize_max as rm
import src.base.rotate_image as roi
import src.base.rotate_points as rp
import src.display.display_images as di

debug_display_all_tps = False
debug_display_filtered_tps = False
debug_display_final_gcps = False

def find_gcps(dem_old, dem_new,
              ortho_old, ortho_new,
              transform,
              resolution, bounding_box,
              rotation,
              mask=None,
              min_conf=0.75,
              cell_size = 2500,
              no_data_value=-9999):

    mask[ortho_old >= 254] = 0
    mask[ortho_old == no_data_value] = 0

    ortho_old[ortho_old == 255] = 0

    if ortho_old.shape != mask.shape:
        print(ortho_old.shape, mask.shape)
        raise ValueError("Ortho and mask have different shapes")
    if dem_old.shape != ortho_old.shape:
        print(dem_old.shape, ortho_old.shape)
        raise ValueError("historic DEM and ortho have different shapes")
    if dem_new.shape != ortho_new.shape[-2:]:
        print(dem_new.shape, ortho_new.shape)
        raise ValueError("modern DEM and ortho have different shapes")

    # resize the old ortho
    ortho_old_resized, s_x_old, s_y_old = rm.resize_max(ortho_old, max_size=20000)
    mask_resized, _, _ = rm.resize_max(mask, max_size=20000)

    # rotate the old ortho
    ortho_old_rotated, rot_mat = roi.rotate_image(ortho_old_resized, rotation,
                                         return_rot_matrix=True)
    mask_rotated = roi.rotate_image(mask_resized, rotation)

    # resize the other images
    ortho_new_resized, s_x_new, s_y_new = ri.resize_image(ortho_new,
                                                          ortho_old_rotated.shape,
                                                          return_scale_factor=True)

    # find tie points
    tpd = ftp.TiePointDetector('lightglue', min_conf=min_conf)

    # find the number of cells in the x and y direction
    n_x = int(np.ceil(ortho_old_rotated.shape[1] / cell_size))
    n_y = int(np.ceil(ortho_old_rotated.shape[0] / cell_size))

    all_tps = []
    all_conf = []

    # iterate over the cells
    for i in range(n_x):
        for j in range(n_y):
            # get the cell
            x_min = i * cell_size
            x_max = min((i + 1) * cell_size, ortho_old_rotated.shape[1])
            y_min = j * cell_size
            y_max = min((j + 1) * cell_size, ortho_old_rotated.shape[0])

            # get the cell from the new ortho, old ortho and mask
            cell_new = ortho_new_resized[:, y_min:y_max, x_min:x_max]
            cell_old = ortho_old_rotated[y_min:y_max, x_min:x_max]
            cell_mask = mask_rotated[y_min:y_max, x_min:x_max]

            # check if the cell is masked completely
            if np.sum(cell_mask) == 0:
                print("completely masked", i, j)
                continue

            # check if the cell is empty
            if np.sum(cell_old) == 0:
                print("Empty cell", i, j)
                continue


            # find the tie points
            tpd = ftp.TiePointDetector('lightglue', min_conf=min_conf)
            tps, conf = tpd.find_tie_points(cell_new, cell_old,
                                            mask2=cell_mask)

            print(f"Found {tps.shape[0]} tie points in cell {i}, {j}")

            if False:
                di.display_images([cell_new, cell_old],
                              tie_points=tps, tie_points_conf=conf)

            # add the cell offset
            tps[:, 0] += x_min
            tps[:, 1] += y_min
            tps[:, 2] += x_min
            tps[:, 3] += y_min

            all_tps.append(tps)
            all_conf.append(conf)

    # concatenate the tie points
    tps = np.concatenate(all_tps)
    conf = np.concatenate(all_conf)

    print(tps.shape[0])

    # find tie points
    tps, conf = tpd.find_tie_points(ortho_new_resized, ortho_old_rotated,
                              mask2=mask_rotated)

    print(f"Found {tps.shape[0]} tie points")

    if tps.shape[0] == 0:
        raise ValueError("No tie points found")

    # scale points back
    tps[:,0] /= s_x_new
    tps[:,1] /= s_y_new

    # rotate points back
    tps[:,2:4] = rp.rotate_points(tps[:,2:4], rot_mat,
                                         invert=True)

    # scale old points back
    tps[:,2] /= s_x_old
    tps[:,3] /= s_y_old

    print(np.amin(tps[:, 0]), np.amax(tps[:, 0]))
    print(np.amin(tps[:, 1]), np.amax(tps[:, 1]))
    print(np.amin(tps[:, 2]), np.amax(tps[:, 2]))
    print(np.amin(tps[:, 3]), np.amax(tps[:, 3]))

    # init variables for tp selection
    selected_tps, selected_conf = [], []
    selection_mask = np.zeros_like(ortho_old)
    selection_radius = 150  # in m

    # order the tie points by confidence
    order = np.argsort(conf)[::-1]
    tps = tps[order]

    # Precompute a circular mask for the selection radius
    y_grid, x_grid = np.ogrid[-selection_radius:selection_radius + 1, -selection_radius:selection_radius + 1]
    circular_mask = x_grid ** 2 + y_grid ** 2 <= selection_radius ** 2

    # iterate over the tie points
    for i in range(tps.shape[0]):
        tp = tps[i]

        x, y = int(tp[2]), int(tp[3])

        # Check if the tie point falls within an already selected region
        if selection_mask[y, x] == 1:
            continue

        # append the tie point to the selected list
        selected_tps.append(tp)
        selected_conf.append(conf[i])

        # Update the selection mask
        y_min = max(0, y - selection_radius)
        y_max = min(selection_mask.shape[0], y + selection_radius + 1)
        x_min = max(0, x - selection_radius)
        x_max = min(selection_mask.shape[1], x + selection_radius + 1)

        # Extract the corresponding subregion of the selection mask
        mask_subregion = selection_mask[y_min:y_max, x_min:x_max]

        # Calculate the corresponding slice for circular_mask
        circular_submask = circular_mask[
                           max(0, selection_radius - y): selection_radius * 2 + 1 - max(0, y + selection_radius + 1 -
                                                                                        selection_mask.shape[0]),
                           max(0, selection_radius - x): selection_radius * 2 + 1 - max(0, x + selection_radius + 1 -
                                                                                        selection_mask.shape[1])
                           ]

        # Automatically fix shape mismatches by resizing circular_submask
        if mask_subregion.shape != circular_submask.shape:
            print(
                f"Fixing shape mismatch: mask_subregion={mask_subregion.shape}, circular_submask={circular_submask.shape}")
            fixed_circular_submask = np.zeros_like(mask_subregion, dtype=bool)
            min_rows = min(mask_subregion.shape[0], circular_submask.shape[0])
            min_cols = min(mask_subregion.shape[1], circular_submask.shape[1])
            fixed_circular_submask[:min_rows, :min_cols] = circular_submask[:min_rows, :min_cols]
            circular_submask = fixed_circular_submask

        # Apply the circular mask to the subregion
        mask_subregion |= circular_submask

    # create np array from list
    tps = np.array(selected_tps)

    #di.display_images([ortho_new, ortho_old], tie_points=tps, tie_points_conf=selected_conf)

    print(tps.shape[0], "tie points left after filtering")

    # get the modern elevation for the tie points
    x_coords_new = tps[:, 0].astype(int)
    y_coords_new = tps[:, 1].astype(int)
    elevations_new = dem_new[y_coords_new, x_coords_new]
    elevations_new = elevations_new.reshape(-1, 1)

    # get the old elevation for the tie points
    x_coords_old = tps[:, 2].astype(int)
    y_coords_old = tps[:, 3].astype(int)
    elevations_old = dem_old[y_coords_old, x_coords_old]
    elevations_old = elevations_old.reshape(-1, 1)

    # assure transform is a numpy array
    if type(transform) != np.ndarray:
        transform = np.array(transform).reshape(3, 3)

    # get absolute coords for the tie points
    ones = np.ones((tps.shape[0], 1))
    abs_px = np.hstack([tps[:, 0:2], ones])
    absolute_coords = abs_px @ transform.T[:, :2]


    # convert the old coords to relative coords
    px_coords = tps[:, 2:4]
    rel_coords = px_coords * resolution
    if type(bounding_box) == list:
        rel_coords[:, 0] = min(bounding_box[0], bounding_box[2]) + rel_coords[:, 0]
        rel_coords[:, 1] = max(bounding_box[1], bounding_box[3]) - rel_coords[:, 1]
    else:
        rel_coords[:, 0] = bounding_box.min[0] + rel_coords[:, 0]
        rel_coords[:, 1] = bounding_box.max[1] - rel_coords[:, 1]

    # Stack the columns into a new array
    stacked_array = np.hstack([px_coords, rel_coords, elevations_old,
                               absolute_coords, elevations_new])
    # convert to pandas dataframe
    df = pd.DataFrame(stacked_array, columns=['x_px', 'y_px',
                                              'x_rel', 'y_rel', 'z_rel',
                                              'x_abs', 'y_abs', 'z_abs'])

    return df


if __name__ == "__main__":

    project_name = "test1"
    path_project_fld = f"/data/ATM/data_1/sfm/agi_projects/{project_name}"

    import os
    output_fld = os.path.join(path_project_fld, "output")
    data_fld = os.path.join(path_project_fld, "data")

    import src.base.calc_bounds as cb
    import src.load.load_image as li
    import src.load.load_rema as lr
    import src.load.load_satellite as ls
    import src.load.load_transform as lt

    old_dem = li.load_image(os.path.join(output_fld, f"{project_name}_dem_relative.tif"))
    transform = lt.load_transform(os.path.join(data_fld, "georef", "transform.txt"), delimiter=",")
    bounds = cb.calc_bounds(transform, old_dem.shape)

    new_dem = lr.load_rema(bounds)
    old_ortho = li.load_image(os.path.join(output_fld, f"{project_name}_ortho_relative.tif"))
    new_ortho, modern_transform = ls.load_satellite(bounds, return_transform=True)

    if len(old_ortho.shape) == 3:
        old_ortho = old_ortho[0, :, :]

    path_cached_rock_mask = os.path.join(data_fld, "georef", "rock_mask.tif")
    rock_mask = li.load_image(path_cached_rock_mask)

    resolution = 0.01

    rotation = np.loadtxt(os.path.join(data_fld, "georef", "best_rot.txt"))
    rotation = float(rotation)

    rock_mask = ri.resize_image(rock_mask, old_ortho.shape)

    df = find_gcps(old_dem, new_dem, old_ortho, new_ortho, modern_transform,
              resolution, bounds, rotation=rotation, mask=rock_mask)

