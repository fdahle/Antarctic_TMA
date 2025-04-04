import os
import glob
import pandas as pd
import src.base.calc_bounds as cb
import src.base.resize_image as ri
import src.load.load_image as li
import src.load.load_rema as lr
import src.load.load_rock_mask as lrm
import src.dem.estimate_dem_quality as edq

PATH_PROJECTS_FOLDER = "/data/ATM/data_1/sfm/agi_projects"


def update_quality_table(project_name, conn=None):

    print("Update quality table for", project_name)

    # get old dem path
    output_dir = os.path.join(PATH_PROJECTS_FOLDER, project_name, "output")
    pattern = os.path.join(output_dir, f"{project_name}_dem_absolute_*.tif")
    lst_dem_c = glob.glob(pattern)
    if len(lst_dem_c) == 0:
        print("No corrected dem found")
        return
    path_corrected_dem = lst_dem_c[0]

    # get the old and modern dem for the project
    dem_old, transform_old = li.load_image(path_corrected_dem, return_transform=True)

    # get bounds from dem and transform
    bounds_old = cb.calc_bounds(transform_old, dem_old.shape)

    # get modern elevation data
    dem_modern, transform_modern = lr.load_rema(bounds_old, return_transform=True)

    # reshape the old dem to the modern dem
    dem_old = ri.resize_image(dem_old, dem_modern.shape)

    # first quality for the whole dem
    quality_all = edq.estimate_dem_quality(dem_modern, dem_old)
    print("  Quality all")
    print("  ", quality_all)

    # get the mask
    mask = lrm.load_rock_mask(bounds_old, 10, mask_buffer=11)

    # apply the mask
    quality_mask = edq.estimate_dem_quality(dem_modern, dem_old, mask)
    print("  Quality mask")
    print("  ", quality_mask)

    # apply slope as well
    quality_slope = edq.estimate_dem_quality(dem_modern, dem_old, mask,
                                             max_slope=30,
                                             slope_transform=transform_modern)
    print("  Quality slope")
    print("  ", quality_slope)

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

    for idx, row in finished_projects.iterrows():
        project_name = row['project']
        update_quality_table(project_name)