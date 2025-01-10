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
              mask_old=None, mask_new=None,
              min_conf=0.9,
              cell_size = 2500,
              no_data_value=-9999):


    if dem_old.shape != ortho_old.shape:
        print(dem_old.shape, ortho_old.shape)
        raise ValueError("historic DEM and ortho have different shapes")
    if dem_new.shape != ortho_new.shape[-2:]:
        print(dem_new.shape, ortho_new.shape)
        raise ValueError("modern DEM and ortho have different shapes")

    # resize the old ortho
    ortho_old_resized, s_x_old, s_y_old = rm.resize_max(ortho_old, max_size=20000)

    ortho_old_resized[ortho_old_resized == 255] = 0

    # create lists that will contain tps and conf of each rotation
    tps_per_rotation = []
    conf_per_rotation = []
    rot_mats = []
    scale_factors = []

    # we want to try out to nudge the rotation a bit
    for nudge in [0, 10, -10]:

        # create adapted rotation
        adapted_rotation = rotation + nudge
        adapted_rotation = adapted_rotation % 360

        # rotate the old ortho
        ortho_old_rotated, rot_mat = roi.rotate_image(ortho_old_resized, adapted_rotation,
                                             return_rot_matrix=True)

        # save rotation matrix
        rot_mats.append(rot_mat)

        # resize and rotate mask
        if mask_old is not None:
            mask_old_resized, _, _ = rm.resize_max(mask_old, max_size=20000)
            mask_old_rotated = roi.rotate_image(mask_old_resized, adapted_rotation)
        else:
            mask_old_rotated = None

        # resize the other images
        ortho_new_resized, s_x_new, s_y_new = ri.resize_image(ortho_new,
                                                              ortho_old_rotated.shape,
                                                              return_scale_factor=True)
        scale_factors.append((s_x_new, s_y_new))

        if mask_new is not None:
            mask_new_resized = ri.resize_image(mask_new, ortho_old_rotated.shape)
        else:
            mask_new_resized = None

        # find the number of cells in the x and y direction
        n_x = int(np.ceil(ortho_old_rotated.shape[1] / cell_size))
        n_y = int(np.ceil(ortho_old_rotated.shape[0] / cell_size))

        all_tps = []
        all_conf = []

        # new tie point detector with higher required confidence
        tpd = ftp.TiePointDetector('lightglue', min_conf=min_conf, tp_type=float)

        # iterate over the cells
        for i in range(n_x):
            for j in range(n_y):
                # get initial cell boundaries
                x_min = i * cell_size
                x_max = min((i + 1) * cell_size, ortho_old_rotated.shape[1])
                y_min = j * cell_size
                y_max = min((j + 1) * cell_size, ortho_old_rotated.shape[0])

                # get initial old cell
                cell_old = ortho_old_rotated[y_min:y_max, x_min:x_max]
                cell_mask_old = mask_old_rotated[y_min:y_max, x_min:x_max]

                if np.sum(cell_mask_old) == 0:
                    #print("Old cell completely masked", i, j)
                    continue

                if np.sum(cell_old) == 0:
                    #print("Old cell empty", i, j)
                    continue

                # Find the number of leading/trailing rows and columns that are zero
                top_zeros = np.argmax(np.any(cell_old != 0, axis=1))  # First non-zero row
                bottom_zeros = np.argmax(np.any(cell_old[::-1] != 0, axis=1))  # First non-zero row from the bottom
                left_zeros = np.argmax(np.any(cell_old != 0, axis=0))  # First non-zero column
                right_zeros = np.argmax(np.any(cell_old[:, ::-1] != 0, axis=0))  # First non-zero column from the right

                if top_zeros > bottom_zeros:
                    y_min += top_zeros
                    y_max += top_zeros
                elif bottom_zeros > top_zeros:
                    y_min -= bottom_zeros
                    y_max -= bottom_zeros
                else:
                    # do nothing
                    pass

                if left_zeros > right_zeros:
                    x_min += left_zeros
                    x_max += left_zeros
                elif right_zeros > left_zeros:
                    x_min -= right_zeros
                    x_max -= right_zeros
                else:
                    # do nothing
                    pass

                # get the new cells with adapted values
                cell_old = ortho_old_rotated[y_min:y_max, x_min:x_max]
                cell_mask_old = mask_old_rotated[y_min:y_max, x_min:x_max]
                cell_new = ortho_new_resized[:, y_min:y_max, x_min:x_max]
                cell_mask_new = mask_new_resized[y_min:y_max, x_min:x_max]

                # check if the cell is masked completely
                if np.sum(cell_mask_new) == 0:
                    #print("New cell completely masked", i, j)
                    continue

                # find the tie points
                tps, conf = tpd.find_tie_points(cell_new, cell_old,
                                                mask1=cell_mask_new,
                                                mask2=cell_mask_old,
                                                mask_final_tps=True)

                print(f"Found {tps.shape[0]} tie points in cell {i}, {j}")

                if False:
                    di.display_images([cell_new, cell_old],
                                      overlays=[cell_mask_new, cell_mask_old],
                                        tie_points=tps, tie_points_conf=conf)

                # skip if no tie points were found
                if tps.shape[0] == 0:
                    continue

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

        print(f"Found {tps.shape[0]} tie points for nudged rotation {adapted_rotation}")

        if tps.shape[0] == 0:
            tps_per_rotation.append(np.zeros((0, 4)))
            conf_per_rotation.append(np.zeros(0))
        else:
            tps_per_rotation.append(tps)
            conf_per_rotation.append(conf)

    # get rotation with most tps
    max_idx = np.argmax([tps.shape[0] for tps in tps_per_rotation])
    tps = tps_per_rotation[max_idx]
    conf = conf_per_rotation[max_idx]
    rot_mat = rot_mats[max_idx]
    s_x_new, s_y_new = scale_factors[max_idx]

    print("Final tie points:", tps.shape[0])

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

    print(tps.shape[0], "tie points left after filtering")

    if tps.shape[0] == 0:
        raise ValueError("No tie points left after filtering")

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

