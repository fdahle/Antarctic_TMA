import os.path
import shutil

import base.connect_to_db as ctd
import base.print_v as p

import sfm_modelling.sub.create_project_structure as cps
import sfm_modelling.sub.execute_mm_cmd as emc

# basic settings
base_fld = "/data_1/ATM/data_1/sfm/"
project_name = "TEST"
resume = True  # can we resume the project
overwrite = False  # if False, we check for existing folders

# input settings
input_type = "ids"  # can be 'ids' or 'area'
input_ids = []
input_area = [776400, -1994000, 749100, -193800]  # min_x, min_y, max_x, max_y
input_directions = []
flight_path_filter = [1846]
use_existing_resampled = False  # should we copy already existing resampled images?

# command settings
commands = ["Tapioca_own", "Schnaps"]
commands = ["Tapas"]

# some general project settings
camera_name = "1983"
image_pattern = "OIS*.*tif"

# fld settings
project_base_fld = base_fld + "projects/"


def create_model():
    # set the actual project folder
    project_fld = project_base_fld + project_name

    # get which images do we need
    sql_string = "SELECT image_id FROM images_extracted WHERE " + \
                 "ST_Intersects(ST_SetSRID(footprint_exact, 3031), " \
                 f"ST_MakeEnvelope({input_area[1]}, {input_area[0]}, {input_area[3]}, {input_area[2]}, 3031))"
    data = ctd.get_data_from_db(sql_string, catch=False)

    # filter by flightpath if specified
    if len(flight_path_filter) > 0:
        data = data[data['image_id'].apply(lambda x: int(x[2:6])).isin(flight_path_filter)]

    # get the input ids
    input_ids = data['image_id'].tolist()

    if len(input_ids) == 0:
        p.print_v("There are no input images", color="red")
        exit()

    # following stuff only needs to be done for new projects
    if resume is False or os.path.isdir(project_fld) is False:

        # check for overwrite
        if overwrite is False and os.path.isdir(project_fld):
            p.print_v(f"Project at {project_fld} is already existing!", color="red")
            exit()

        # remove all content from the folder
        if overwrite and os.path.isdir(project_fld):
            shutil.rmtree(project_fld)

        # check if we need to create a new folder:
        if os.path.isdir(project_fld) is False:
            os.mkdir(project_fld)

        # create folder structure if it is a new project
        cps.create_project_structure(project_fld, input_ids, camera_name,
                                     copy_resampled=use_existing_resampled,
                                     verify=True)

    ###
    # execute micmac commands in order
    ###

    if "ReSampFid" in commands:
        # args for ReSampFid
        res_args = {
            'ImagePattern': "*.*tif",
            'ScanResolution': 0.025
        }

        # resample the images
        emc.execute_mm_cmd("ReSampFid", res_args, project_fld,
                           image_ids=input_ids,
                           save_stats=save_stats, stats_folder=stats_folder,
                           delete_temp_files=delete_temp_files,
                           print_output=print_output, print_orig_errors=print_orig_errors)

    if "Tapioca_own" in commands:
        # own methods of tie point matching
        emc.execute_mm_cmd("Tapioca_own", "", project_fld)

    if "HomolFilterMasq" in commands:
        # args for HomolFilterMasq
        hom_args = {
            'ImagePattern': image_pattern
        }

        # filtering on tie points
        emc.execute_mm_cmd("HomolFilterMasq", hom_args, project_fld,
                           save_stats=save_stats, stats_folder=stats_folder,
                           delete_temp_files=delete_temp_files,
                           print_output=print_output, print_orig_errors=print_orig_errors)

    if "Schnaps" in commands:
        # args for Schnaps
        sch_args = {
            'ImagePattern': image_pattern,
            'ExpTxt': 1
        }

        # tie point reduction tool
        emc.execute_mm_cmd("Schnaps", sch_args, project_fld,
                           save_stats=save_stats, stats_folder=stats_folder,
                           delete_temp_files=delete_temp_files,
                           print_output=print_output, print_orig_errors=print_orig_errors)

        # copy the files without tie-points back in their folders, as otherwise tapas raises error
        if os.path.isfile(project_fld + "/Schnaps_poubelle.txt"):
            with open(project_fld + "/Schnaps_poubelle.txt") as f:
                for line in f:
                    # copy invalid images
                    shutil.move(project_fld + "/" + line.strip(), project_fld + "/images/" + line.strip())
        else:
            # create empty schnaps file to state that we used schnaps
            with open(project_fld + "/Schnaps_poubelle.txt") as w:
                pass

    if "Tapas" in commands:

        # the input folder for tapas changes if Schnaps is used
        if os.path.isfile(project_fld + "/Schnaps_poubelle.txt"):
            sh = "Homol_mini"
        else:
            sh = "Homol"

        # args for Tapas
        tap_args = {
            'DistortionModel': "RadialBasic",
            'ExpTxt': 1,
            'ImagePattern': image_pattern,
            'LibFoc': 0,
            'Out': "Relative",
            'SH': sh
        }

        # compute relative orientation
        emc.execute_mm_cmd("Tapas", tap_args, project_fld,
                           save_stats=save_stats, stats_folder=stats_folder,
                           delete_temp_files=delete_temp_files,
                           print_output=print_output, print_orig_errors=print_orig_errors)

    if "AperiCloud" in commands:

        # args for AperiCloud
        ape_args = {
            'ImagePattern': image_pattern,
            'Orientation': ""
        }

        # visualize relative orientation
        emc.execute_mm_cmd("AperiCloud", ape_args, project_fld,
                           save_stats=save_stats, stats_folder=stats_folder,
                           delete_temp_files=delete_temp_files,
                           print_output=print_output, print_orig_errors=print_orig_errors)

    if "Malt" in commands:
        # args for Malt
        mal_args = {
            'ImagePattern': image_pattern,
            'Mode': "",
            'Orientation': ""
        }

        # compute DEM
        emc.execute_mm_cmd("Malt", mal_args, project_fld,
                           save_stats=save_stats, stats_folder=stats_folder,
                           delete_temp_files=delete_temp_files,
                           print_output=print_output, print_orig_errors=print_orig_errors)


if __name__ == "__main__":
    create_model()
