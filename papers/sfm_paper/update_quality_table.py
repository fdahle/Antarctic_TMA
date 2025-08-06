import os
import glob
import Metashape
import pandas as pd
import src.base.connect_to_database as ctd
import src.base.calc_bounds as cb
import src.base.resize_image as ri
import src.load.load_image as li
import src.load.load_rema as lr
import src.load.load_rock_mask as lrm
import src.dem.estimate_dem_quality as edq
import src.sfm_agi.snippets.get_project_quality as gpc

PATH_PROJECTS_FOLDER = "/data/ATM/data_1/sfm/agi_projects"

conn = ctd.establish_connection()
update_project = False

def update_quality_table(project_name, conn=None):

    print("Update quality table for", project_name)


    # get old dem path
    output_dir = os.path.join(PATH_PROJECTS_FOLDER, project_name, "output")
    path_old_dem = os.path.join(output_dir, project_name + "_dem_absolute.tif")
    output_dir_corr = "/data/ATM/data_1/papers/paper_sfm/finished_dems_corrected2/"

    pattern = os.path.join(output_dir_corr, f"{project_name}_dem_corrected.tif")
    print(pattern)
    lst_dem_c = glob.glob(pattern)
    if len(lst_dem_c) == 0:
        print("No corrected dem found")
        return
    path_corrected_dem = lst_dem_c[0]

    # get the old and modern dem for the project
    dem_old, transform_old = li.load_image(path_old_dem, return_transform=True)
    dem_corrected, transform_corrected = li.load_image(path_corrected_dem, return_transform=True)

    # get bounds from dem and transform
    bounds_old = cb.calc_bounds(transform_old, dem_old.shape)
    bounds_corrected= cb.calc_bounds(transform_corrected, dem_corrected.shape)

    # get modern elevation data
    dem_modern, transform_modern = lr.load_rema(bounds_old, return_transform=True)
    dem_modern_c, transform_modern_c = lr.load_rema(bounds_corrected,  return_transform=True)

    # reshape the old dem to the modern dem
    dem_old = ri.resize_image(dem_old, dem_modern.shape)
    dem_corrected = ri.resize_image(dem_corrected, dem_modern_c.shape)

    # first quality for the whole dem
    quality_all = edq.estimate_dem_quality(dem_modern, dem_old)
    print("  Quality all")
    print("  ", quality_all)

    quality_all_corrected = edq.estimate_dem_quality(dem_modern_c, dem_corrected)
    print("  Quality all corrected")
    print("  ", quality_all_corrected)

    # get the mask
    mask = lrm.load_rock_mask(bounds_old, 10, mask_buffer=11)
    mask_corrected = lrm.load_rock_mask(bounds_corrected, 10, mask_buffer=11)

    # apply the mask
    quality_mask = edq.estimate_dem_quality(dem_modern, dem_old, mask)
    print("  Quality mask")
    print("  ", quality_mask)

    quality_mask_corrected = edq.estimate_dem_quality(dem_modern_c, dem_corrected,
                                                      mask_corrected)
    print("  Quality mask corrected")
    print("  ", quality_mask_corrected)

    # apply slope as well
    quality_slope = edq.estimate_dem_quality(dem_modern, dem_old, mask,
                                             max_slope=30,
                                             slope_transform=transform_modern)
    print("  Quality slope")
    print("  ", quality_slope)

    quality_slope_corrected = edq.estimate_dem_quality(dem_modern_c, dem_corrected,
                                                       mask_corrected,
                                                       max_slope=30,
                                                       slope_transform=transform_modern_c)
    print("  Quality slope corrected")
    print("  ", quality_slope_corrected)

    # load the project and get the chunk
    if update_project:
        project_dir = os.path.join(PATH_PROJECTS_FOLDER, project_name)
        project_psx_path = os.path.join(project_dir, f"{project_name}.psx")

        doc = Metashape.Document(read_only=True)  # noqa
        doc.open(project_psx_path, ignore_lock=True)
        chunk = doc.chunks[0]

        # get nr and quality of gcps
        project_dict = gpc.get_project_quality(chunk)
        print("  Project Quality")
        print("  ", project_dict)

    # Mapping of quality keys to SQL columns per quality type
    quality_prefix_map = {
        "all": quality_all,
        "mask": quality_mask,
        "slope": quality_slope,
        "c_all": quality_all_corrected,
        "c_mask": quality_mask_corrected,
        "c_slope": quality_slope_corrected
    }
    key_map = {
        "mean_difference": "diff_mean",
        "std_difference": "diff_std",
        "mean_difference_abs": "diff_abs_mean",
        "difference_abs_std": "diff_abs_std",
        "median_difference": "diff_median",
        "median_difference_abs": "diff_abs_median",
        "rmse": "rmse",
        "mae": "mae",
        "mad": "mad",
        "correlation": "corr"
    }

    sql_string = f"UPDATE sfm_projects3 SET project_name='{project_name}', "

    if update_project:
        for key in project_dict.keys():
            value = project_dict[key]
            sql_string += f"{key}={value}, "

    for prefix, quality in quality_prefix_map.items():
        for key, column_suffix in key_map.items():
            colname = f"{prefix}_{column_suffix}"
            value = quality[key]
            sql_string += f"{colname}={value}, "

    sql_string = sql_string.rstrip(", ")  # Remove trailing comma
    sql_string += f" WHERE project_name='{project_name}';"

    ctd.execute_sql(sql_string, conn, add_timestamp=False)

def get_all_finished():

    rows = []
    # iterate over all project folders
    for entry in os.listdir(PATH_PROJECTS_FOLDER):
        project_fld = os.path.join(PATH_PROJECTS_FOLDER, entry)

        # get project name
        project_name = project_fld.split("/")[-1]

        # define the output folder
        output_dir = os.path.join(project_fld, "output")

        # check if the project was finished with the absolute ortho
        if not os.path.exists(os.path.join(output_dir, project_name + "_ortho_absolute.tif")):
            continue

        # add the project to the output data
        rows.append({"project": project_name})

    finished_projects = pd.DataFrame(rows)
    return finished_projects

if __name__ == "__main__":
    finished_projects = get_all_finished()

    from tqdm import tqdm

    for idx, row in tqdm(finished_projects.iterrows(), total=len(finished_projects)):
        project_name = row['project']

        if project_name != "northeast_glacier":
            continue

        update_quality_table(project_name, conn=conn)