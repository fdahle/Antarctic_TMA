import os
import subprocess
import glob
import json

from tqdm import tqdm

debug_print = False
debug_print_errors = True

def gcpbascule(project_folder, m_args,
            save_stats=True, stats_folder="stats",
            delete_temp_files=True,
            print_output=False, print_orig_errors=False):

    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder

    required_args = ["ImagePattern", "InputOrientation", "OutputOrientation",
                     "FileGroundControlPoints", "FileImageMeasurements"]
    allowed_args = ["ImagePattern", "InputOrientation", "OutputOrientation",
                    "FileGroundControlPoints", "FileImageMeasurements", "L1", "CPI",
                    "ShowU", "ShowD", "PatNLD", "NLDDegX", "NLDDegY", "NLDDegZ", "NLFR",
                    "NLShow"]

    # function to check if the input parameters are valid
    def __validate_input():

        # check project folder
        assert os.path.isdir(project_folder), \
            f"'{project_folder}' is not a valid path to a folder"
        if save_stats:
            assert os.path.isdir(stats_folder), \
                f"stats folder is missing at '{stats_folder}'"

        # check if we have the required arguments
        for r_arg in required_args:
            assert r_arg in m_args, f"{r_arg} is a required argument"

        # check if only allowed arguments were used
        for arg in m_args:
            assert arg in allowed_args, f"{arg} is not an allowed argument"

        # get the images we are working with
        _lst_images = []
        for _elem_ in glob.glob(project_folder + "/" + m_args["ImagePattern"]):
            _img_name = _elem_.split("/")[-1].split(".")[0]
            _lst_images.append(_img_name)

    # validation the input parameters
    __validate_input()

    # here we define the error messages that can happen
    error_dict = {
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {}

    # the actual calling of gcpbascule
    shell_string = f'mm3d GCPBascule "{m_args["ImagePattern"]}" {m_args["InputOrientation"]} ' \
                   f'{m_args["OutputOrientation"]} {m_args["FileGroundControlPoints"]} ' \
                   f'{m_args["FileImageMeasurements"]}'

    # add the arguments to the shell string
    for key, val in m_args.items():

        # required arguments are called extra
        if key in required_args:
            continue

        shell_string = shell_string + " " + str(key) + "=" + str(val)

    print("Start with GCPBascule")
    print(shell_string)


if __name__ == "__main__":

    temp_folder = "/data_1/ATM/data/sfm/projects/func_test"

    img_pat = ""
    orientation_in = ""
    orientation_out = ""
    gcp_file = ""
    image_measurement_file = ""

    args = {
        "ImagePattern": img_pat,
        "InputOrientation": orientation_in,
        "OutputOrientation": orientation_out,
        "FileGroundControlPoints": gcp_file,
        "FileImageMeasurements": image_measurement_file
    }

    gcpbascule(
        project_folder=temp_folder)