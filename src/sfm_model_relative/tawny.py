import os
import subprocess
import json
import glob

from tqdm import tqdm

import stitch_micmac_tiles as smt

debug_print = False
debug_print_errors = True


def tawny(project_folder, m_args,
          save_stats=True, stats_folder="stats",
          merge_files=False,
          delete_temp_files=True,
          print_output=False, print_orig_errors=False):
    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder

    required_args = ["DataDirectory"]
    allowed_args = ["DataDirectory", "RadiomEgal", "DEq", "DEqXY", "AddCste", "DegRap", "DegRapXY",  # noqa
                    "RGP", "DynG", "ImPrio", "SzV", "CorThr", "NbPerlm", "L1F", "SatThresh", "Out"]  # noqa

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

        # check Data directory for validity
        assert os.path.isdir(project_folder + "/" + m_args["DataDirectory"]), \
            f"'{project_folder + '/' + m_args['DataDirectory']}' is not a valid path to a folder"

        # check some args for validity
        if "RadiomEgal" in m_args:
            assert m_args["RadiomEgal"] in [0, 1], \
                "The value for RadiomEgal is invalid."

    # validation the input parameters
    __validate_input()

    # here we define the error messages that can happen
    error_dict = {
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {
        "images": {}
    }

    # the actual calling of tawny
    shell_string = f'mm3d Tawny "{m_args["DataDirectory"]}"'

    # add the arguments to the shell string
    for key, val in m_args.items():

        # required arguments are called extra
        if key in required_args:
            continue

        shell_string = shell_string + " " + str(key) + "=" + str(val)

    print("Start with Tawny")
    print(shell_string)

    with subprocess.Popen(["/bin/bash", "-i", "-c", shell_string],
                          cwd=project_folder,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True
                          ) as p:

        for stdout_line in tqdm(p.stdout):

            # don't print the comments
            if "*" in stdout_line[0:2]:
                if print_output is False:
                    continue

            # don't print whitespace
            if stdout_line.isspace():
                if print_output is False:
                    continue

            # don't print unnecessary text
            if stdout_line.startswith("---------------------"):
                if print_output is False:
                    continue

            # skip unnecessary text part I
            suffixes = ["matches.\n"]
            if stdout_line.endswith(tuple(suffixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part II
            prefixes = ["KBOX", "OUT", " IN", " Radiom Max"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part III
            mifixes = ["Porto"]
            if any(x in stdout_line for x in mifixes):
                if print_output is False:
                    continue

            # get which images are used
            if stdout_line.startswith("Image"):
                stdout_line = stdout_line.replace("\n", "")
                splits = stdout_line.split(" ")
                stats["images"][splits[1]] = {
                    "label": splits[3],
                    "prio": stdout_line.split("=")[-1]
                }
                if print_output is False:
                    continue

            # catch errors
            if stdout_line.startswith("| ") and stdout_line.count('|') == 1:

                for elem in error_dict.keys():
                    if elem in stdout_line:
                        error_msg = error_dict[elem]

                if print_orig_errors:
                    print(stdout_line, end="")
                continue

            # end in case of error
            if stdout_line.startswith("Bye  (press enter)"):
                print(ValueError(error_msg))

                p.kill()
                exit()

            # the last resort: everything we didn't catch before is printed here
            print(stdout_line, end="")

        if save_stats:
            with open(stats_folder + "/tawny.json", "w") as outfile:
                json.dump(stats, outfile, indent=4)

    # merge the files if wished
    if merge_files:

        if "Out" not in m_args:
            out_file = "Orthophotomosaic.tif"
        else:
            out_file = m_args["Out"]

        out_file_short = out_file.split(".")[0]

        # get the output images
        _lst_images = []
        for _elem_ in glob.glob(project_folder + "/" + m_args["DataDirectory"] + "/" + out_file_short + "_Tile*.tif"):
            _lst_images.append(_elem_)

        merged = smt.stitch_micmac_tiles(_lst_images)

        from PIL import Image, TiffImagePlugin
        im = Image.fromarray(merged)
        TiffImagePlugin.WRITE_LIBTIFF = True
        im.save(project_folder + "/" + m_args["DataDirectory"] + "/" + out_file_short + ".tif",
                compression="tiff_lzw")
        TiffImagePlugin.WRITE_LIBTIFF = False

    # delete temporary files from the folder
    if delete_temp_files:

        if os.path.isfile(project_folder + "/mm3d-LogFile.txt"):
            os.remove(project_folder + "/mm3d-LogFile.txt")
        for file in os.listdir(project_folder):
            filename = os.fsdecode(file)
            if filename.startswith("MM-Error-") and filename.endswith(".txt"):
                os.remove(project_folder + "/" + filename)


if __name__ == "__main__":
    temp_folder = "/data_1/ATM/data/sfm/" \
                  "projects/final_test"

    data_dir = "Ortho-MEC-Malt"
    radiom = 0
    out_name = "Orthophotomosaic.tif"

    args = {
        "DataDirectory": data_dir,
        "RadiomEgal": radiom,
        "Out": out_name
    }

    tawny(
        project_folder=temp_folder,
        m_args=args,
        merge_files=True,
        print_output=debug_print,
        print_orig_errors=debug_print_errors
    )
