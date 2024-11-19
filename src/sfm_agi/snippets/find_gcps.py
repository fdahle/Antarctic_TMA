import numpy as np
import pandas as pd

import src.base.find_tie_points as ftp
import src.base.resize_image as ri
import src.base.rotate_image as roi
import src.base.rotate_points as rp
import src.display.display_images as di

debug_display_all_tps = True
debug_display_filtered_tps = True
debug_display_final_gcps = True

def find_gcps(dem_old, dem_new,
              ortho_old, ortho_new,
              transform,
              resolution, bounding_box,
              best_rot=None,
              mask=None,
              min_conf=0.75,
              cell_size=250,
              no_data_value=-9999):

    # make new ortho grayscale
    ortho_new = np.mean(ortho_new, axis=0)

    # assure dem and ortho have same shape
    if dem_old.shape != ortho_old.shape:
        raise ValueError(f"Old Ortho {ortho_old.shape} and new DEM {dem_old.shape} "
                         f"have different shapes")
    if dem_new.shape != ortho_new.shape:
        raise ValueError(f"new Ortho {ortho_new.shape} and new DEM {dem_new.shape} "
                         "have different shapes")
    if mask.shape != ortho_old.shape:
        raise ValueError(f"Mask {mask.shape} and old Ortho {ortho_old.shape} "
                         f"have different shapes")

    # set old dem to none where no data value is
    dem_old[dem_old == no_data_value] = np.nan

    # resize images to same size (=ortho old)
    dem_old_r = ri.resize_image(dem_old, ortho_old.shape)
    dem_new_r = ri.resize_image(dem_new, dem_old.shape)
    ortho_new_r, s_x, s_y = ri.resize_image(ortho_new, ortho_old.shape,
                                            return_scale_factor=True)
    if mask is not None:
        mask_r = ri.resize_image(mask, ortho_old.shape)
    else:
        mask_r = None

    # init the tie point detector
    tpd = ftp.TiePointDetector('lightglue',
                               min_conf_value=min_conf)

    # define rotation values
    if best_rot is not None:
        rotation_values = [best_rot]
    else:
        rotation_values = range(0, 360, 20)

    # init variables for best results
    best_rot_mat = None
    best_tps = np.empty((0, 4))
    best_conf = np.empty((0, 1))

    # try different rotations
    for rot in rotation_values:
        ortho_old_ro, rot_mat = roi.rotate_image(ortho_old, rot, expand=True,
                                                 return_rot_matrix=True)
        if mask is not None:
            mask_ro = roi.rotate_image(mask_r, rot, expand=True)
        else:
            mask_ro = None

        # find tie points between old and new ortho
        tie_points, conf = tpd.find_tie_points(ortho_old_ro, ortho_new_r,
                                               mask1=mask_ro, mask2=mask_r)

        # check if the new tie points are better than the old ones
        if tie_points.shape[0] > best_tps.shape[0]:
            best_tps = tie_points
            best_conf = conf
            best_rot_mat = rot_mat

    # set the best values
    tie_points = best_tps
    conf = best_conf
    rot_mat = best_rot_mat

    if tie_points.shape[0] == 0:
        raise ValueError("No tie points found between the ortho images")
    print("Found", tie_points.shape[0], "tie points for gcps")

    # rotate points back
    tie_points[:,0:2] = rp.rotate_points(tie_points[:,0:2], rot_mat,
                                         invert=True)

    if debug_display_all_tps:
        di.display_images([ortho_old, ortho_new_r],
                           image_types=["gray", "gray"],
                           tie_points=tie_points,
                           tie_points_conf=conf,
                           overlays=[dem_old_r, dem_new_r])

    # init variables for tp selection
    selected_tps, selected_conf = [], []
    selection_mask = np.zeros_like(ortho_old)
    selection_radius = 250

    # order the tie points by confidence
    order = np.argsort(conf)[::-1]
    tie_points = tie_points[order]
    conf = conf[order]

    # iterate over the tie points
    for i in range(tie_points.shape[0]):
        tp = tie_points[i]

        x, y = int(tp[0]), int(tp[1])

        # Check if the tie point falls within an already selected region
        if selection_mask[y, x] == 1:
            continue

        # append the tie point to the selected list
        selected_tps.append(tp)
        selected_conf.append(conf[i])

        # Update the mask: Set a circular region around the tie point to 1
        y_grid, x_grid = np.ogrid[:selection_mask.shape[0], :selection_mask.shape[1]]
        distance_from_tp = np.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
        selection_mask[distance_from_tp <= selection_radius] = 1

    # create np array from list
    tie_points = np.array(selected_tps)
    conf = np.array(selected_conf)

    print(tie_points.shape[0], "tie points left after filtering")

    if debug_display_filtered_tps:
        di.display_images([ortho_old, ortho_new_r],
                           image_types=["gray", "gray"],
                           tie_points=tie_points,
                           tie_points_conf=conf,
                           overlays=[dem_old_r, dem_new_r])

    # resize the tie points to the old size
    tie_points[:, 2] = tie_points[:, 2] / s_x
    tie_points[:, 3] = tie_points[:, 3] / s_y

    # get the old elevation for the tie points
    x_coords_old = tie_points[:, 0].astype(int)
    y_coords_old = tie_points[:, 1].astype(int)
    elevations_old = dem_old[y_coords_old, x_coords_old]
    elevations_old = elevations_old.reshape(-1, 1)

    # get the modern elevation for the tie points
    x_coords_new = tie_points[:, 2].astype(int)
    y_coords_new = tie_points[:, 3].astype(int)
    elevations_new = dem_new[y_coords_new, x_coords_new]
    elevations_new = elevations_new.reshape(-1, 1)

    # assure transform is a numpy array
    if type(transform) != np.ndarray:
        transform = np.array(transform).reshape(3, 3)

    # get absolute coords for the tie points
    ones = np.ones((tie_points.shape[0], 1))
    abs_px = np.hstack([tie_points[:, 2:4], ones])
    absolute_coords = abs_px @ transform.T[:, :2]

    # convert the old coords to relative coords
    px_coords = tie_points[:, 0:2]
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

    if debug_display_final_gcps:
        di.display_images([ortho_old, ortho_new, dem_old, dem_new],
                           image_types=["gray", "gray", "dem", "dem"],
                          points=[tie_points[:, 0:2], tie_points[:, 2:4],
                                  tie_points[:, 0:2], tie_points[:, 2:4]],
                          separate_min_max=True)
    exit()
    return df


if __name__ == "__main__":


    project_name = "grf_test"


    # fix all paths
    import os
    PATH_PROJECT_FOLDERS = "/data/ATM/data_1/sfm/agi_projects"
    project_fld = os.path.join(PATH_PROJECT_FOLDERS, project_name)
    output_fld = os.path.join(project_fld, "output")
    data_fld = os.path.join(project_fld, "data")
    output_path_dem_rel = os.path.join(output_fld,
                                       project_name + "_dem_relative.tif")
    output_path_ortho_rel = os.path.join(output_fld,
                                         project_name + "_ortho_relative.tif")
    output_path_conf_rel = os.path.join(output_fld,
                                        project_name + "_confidence_relative.tif")
    transform_path = os.path.join(data_fld, "georef", "transform.txt")
    rock_path = os.path.join(data_fld, "georef", "rock_mask.tif")

    # load all output files
    print("Load output files")
    import src.load.load_image as li
    import src.load.load_transform as lt
    hist_dem = li.load_image(output_path_dem_rel)
    hist_ortho = li.load_image(output_path_ortho_rel)
    hist_ortho = hist_ortho[0,:,:]
    if os.path.isfile(output_path_conf_rel):
        hist_conf = li.load_image(output_path_conf_rel)
    else:
        hist_conf = None
    transform = lt.load_transform(transform_path, delimiter=",")

    # calc bounds
    print("Calc bounds")
    import src.base.calc_bounds as cb
    bounds = cb.calc_bounds(transform, hist_dem.shape)

    # load modern data
    print("Load modern data")
    import src.load.load_rema as lr
    import src.load.load_satellite as ls
    modern_dem = lr.load_rema(bounds)
    modern_ortho, transform_sat = ls.load_satellite(bounds, return_transform=True)

    # create mask
    import src.load.load_rock_mask as lrm
    gcp_mask = np.ones_like(hist_dem)

    if os.path.exists(rock_path):
        rock_mask = li.load_image(rock_path)
    else:
        rock_mask = lrm.load_rock_mask(bounds, 10, mask_buffer=100)

        import src.export.export_tiff as et
        et.export_tiff(rock_mask, rock_path)

    rock_mask = ri.resize_image(rock_mask, hist_dem.shape)
    #di.display_images([modern_ortho], overlays=[modern_dem])

    gcp_mask[rock_mask == 0] = 0
    if hist_conf is not None:
        gcp_mask[hist_conf < 0.8] = 0

    print("Find gcps")
    find_gcps(hist_dem, modern_dem, hist_ortho, modern_ortho,
              transform_sat, 0.001, bounds, mask=gcp_mask)