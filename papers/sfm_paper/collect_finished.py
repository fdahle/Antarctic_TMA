"""collect all finished results and save them in a folder"""

import glob
import os

PATH_PROJECTS_FOLDER = "/data/ATM/data_1/sfm/agi_projects"
PATH_PAPER_FOLDER = "/data/ATM/data_1/papers/paper_sfm"

def collect_finished():

    # iterate over all project folders
    for entry in os.listdir(PATH_PROJECTS_FOLDER):
        project_fld = os.path.join(PATH_PROJECTS_FOLDER, entry)

        # get project name
        project_name = project_fld.split("/")[-1]
        print("Collect data from", project_name)

        # define the output folder
        output_dir = os.path.join(project_fld, "output")

        # check if the project was finished with the absolute ortho
        if not os.path.exists(os.path.join(output_dir, project_name + "_ortho_absolute.tif")):
            continue

        # create link for ortho absolute
        ortho_path_src = os.path.join(output_dir, project_name + "_ortho_absolute.tif")
        ortho_path_dest = os.path.join(PATH_PAPER_FOLDER, "finished_orthos", project_name + "_ortho_absolute.tif")
        #_safe_symlink(ortho_path_src, ortho_path_dest)

        # create link for dem absolute
        dem_path_src = os.path.join(output_dir, project_name + "_dem_absolute.tif")
        dem_path_dest = os.path.join(PATH_PAPER_FOLDER, "finished_dems", project_name + "_dem_absolute.tif")
        #_safe_symlink(dem_path_src, dem_path_dest)

        # get the exact name for dem corrected
        pattern = os.path.join(output_dir, f"{project_name}_dem_absolute_*.tif")
        lst_dem_c = glob.glob(pattern)
        if len(lst_dem_c) == 0:
            print("No corrected dem found")
            continue
        name_corrected_dem = os.path.basename(lst_dem_c[0])

        # create link for dem corrected
        dem_c_path_src = os.path.join(output_dir, name_corrected_dem)
        dem_c_path_dest = os.path.join(PATH_PAPER_FOLDER, "finished_dems_corrected", name_corrected_dem)
        _safe_symlink(dem_c_path_src, dem_c_path_dest)

def _safe_symlink(src, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.lexists(dest):  # covers both real files and broken symlinks
        os.remove(dest)
    os.symlink(src, dest)



if __name__ == "__main__":
    collect_finished()
