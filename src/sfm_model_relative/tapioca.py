import os
import subprocess
import shutil
import fnmatch
import json

from tqdm import tqdm

debug_print = False
debug_print_errors = True


def tapioca(project_folder, m_args,
            save_stats=True, stats_folder="stats",
            delete_temp_files=True,
            print_output=False, print_orig_errors=False):
    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder

    # set the required arguments
    required_args = ["Mode"]
    required_args_all = ["ImagePattern", "ImgSize"]
    required_args_mulscale = ["ImagePattern", "LowResolutionImgSize", "HighResolutionImgSize"]
    required_args_line = ["ImagePattern", "Imgsize", "NumberAdjacentImages"]
    required_args_graph = ["ImagePattern", "ImgSize"]
    required_args_file = ["XMLFile", "Resolution"]

    allowed_args = ["Mode", "ExpTxt", "ByP"]
    additional_args_all = ["ImagePattern", "ImgSize", "PostFix", "Pat2", "Detect",
                           "Match", "NoMax", "NoMin", "NoUnknown"]
    additional_args_mulscale = ["ImagePattern", "LowResolutionImgSize", "HighResolutionImgSize",
                                "PostFix", "NbMinPt", "DLR", "Pat2", "Detect", "Match",
                                "NoMax", "NoMin", "NoUnknown"]
    additional_args_line = ["ImagePattern", "Imgsize", "NumberAdjacentImages", "Jump",
                            "PostFix", "Circ", "ForceAdSupResol", "Detect", "Match",
                            "NoMax", "NoMin", "NoUnknown"]
    additional_args_graph = ["ImagePattern", "ImgSize", "Detect", "MaxPoint",
                             "MinScale", "MaxScale", "NbRequired", "Out", "PrintGraph"]
    additional_args_file = ["XMLFile", "Resolution", "PostFix", "Detect", "Match",
                            "NoMax", "NoMin", "NoUnknown"]

    lst_of_modes = ["All", "MulScale", "Line", "File", "Graph"]

    # extend allowed arguments based on mode
    if m_args["Mode"] == "All":
        required_args = required_args + required_args_all
        allowed_args = allowed_args + additional_args_all
    elif m_args["Mode"] == "MulScale":
        required_args = required_args + required_args_mulscale
        allowed_args = allowed_args + additional_args_mulscale
    elif m_args["Mode"] == "Line":
        required_args = required_args + required_args_line
        allowed_args = allowed_args + additional_args_line
    elif m_args["Mode"] == "File":
        required_args = required_args + required_args_graph
        allowed_args = allowed_args + additional_args_graph
    elif m_args["Mode"] == "Graph":
        required_args = required_args + required_args_file
        allowed_args = allowed_args + additional_args_file

    # function to check if the input parameters are valid
    def __validate_input():

        assert os.path.isdir(project_folder), \
            f"'{project_folder}' is not a valid path to a folder"
        if save_stats:
            assert os.path.isdir(stats_folder), \
                f"stats folder is missing at '{stats_folder}'"

        # check if we have the required arguments
        for _elem in required_args:
            assert _elem in m_args, f"{_elem} is a required argument"

        # check Mode for validity
        assert m_args["Mode"] in lst_of_modes, \
            "The argument for 'Mode' is wrong"

        # check if only allowed arguments were used
        for arg in m_args:
            assert arg in allowed_args, f"{arg} is not an allowed argument"

        # check if we can find images in the image folder that match the pattern
        num_images = 0
        for image_file in os.listdir(project_folder):
            if image_file.endswith(".tif"):
                if fnmatch.fnmatch(image_file, m_args["ImagePattern"]):
                    num_images += 1

        assert num_images >= 2, "At least two images are required for matching"

    __validate_input()

    # here we define the error messages that can happen
    error_dict = {
        "0 matches": "No images did match the provided image pattern"

    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {
        "points": {},
        "combinations": {}
    }

    # required to save the correct stats
    last_id_1, last_id_2 = None, None

    # the actual calling of tapioca
    shell_string = f'mm3d Tapioca '

    # add the arguments to the shell string
    for key, val in m_args.items():

        if key in required_args:
            if key == "ImagePattern":
                shell_string = shell_string + '"' + str(val) + '" '
            else:
                shell_string = shell_string + str(val) + " "
        else:
            shell_string = shell_string + str(key) + "=" + str(val)

    print("Start with Tapioca")
    print(shell_string)

    with subprocess.Popen(["/bin/bash", "-i", "-c", shell_string],
                          cwd=project_folder,
                          stdout=subprocess.PIPE,
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
            suffixes = ["matches.\n", "sift points\n"]
            if stdout_line.endswith(tuple(suffixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part II
            mifixes = ["TestLib  XmlXif", "PastDevlop", "pastis", "Ann ./Pastis", "Sift"]
            if any(x in stdout_line for x in mifixes):
                if print_output is False:
                    continue

            # skip unnecessary text part III
            prefixes = ["BY DICO"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part IV
            prefixes = ["make: ***", "BY FILE", "Cple Init", "Apres Rm Dup"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # get detection method
            if "--- using detecting tool" in stdout_line:
                detecting_method = stdout_line.split(":")[2][:-2]
                stats["detecting_method"] = detecting_method

                if print_output is False:
                    continue

            # get resolution
            if stdout_line.startswith("Ss-RES"):
                splits = stdout_line.split(" ")
                stats["resolution"] = {}
                stats["resolution"]["small"] = splits[2]
                stats["resolution"]["full"] = splits[4].split("=")[-1][:-2]

                if print_output is False:
                    continue

            # get matching method
            if "--- using matching tool" in stdout_line:
                matching_method = stdout_line.split(":")[2][:-2]
                stats["matching_method"] = matching_method

                if print_output is False:
                    continue

            # get the number of points and matches
            if stdout_line.startswith("./Pastis"):
                splits = stdout_line.split(" ")
                img_id_1 = splits[0][37:-8]
                points_1 = splits[2]
                img_id_2 = splits[4][37:-8]
                points_2 = splits[6]
                matches = splits[9]

                stats["points"][img_id_1] = points_1
                stats["points"][img_id_2] = points_2

                if img_id_1 not in stats["combinations"]:
                    stats["combinations"][img_id_1] = {}
                if img_id_2 not in stats["combinations"][img_id_1]:
                    stats["combinations"][img_id_1][img_id_2] = {}

                stats["combinations"][img_id_1][img_id_2]["matches"] = matches

                if print_output is False:
                    continue

            if stdout_line.startswith("OK GLOB"):
                splits = stdout_line.split("/")
                last_id_1 = splits[2][16:-4]
                last_id_2 = splits[3][10:-9]

                if print_output is False:
                    continue

            if stdout_line.startswith("Apres Hom"):
                if last_id_1 is not None and last_id_2 is not None:
                    ap_hom = stdout_line.split(" ")[3][:-2]
                    stats["combinations"][last_id_1][last_id_2]["apres_hom"] = ap_hom

                if print_output is False:
                    continue

            if stdout_line.startswith("Apres Rot"):
                if last_id_1 is not None and last_id_2 is not None:
                    ap_rot = stdout_line.split(" ")[3][:-2]
                    stats["combinations"][last_id_1][last_id_2]["apres_rot"] = ap_rot

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

    print("Finished with Tapioca")

    if save_stats:
        with open(stats_folder + "/tapioca.json", "w") as outfile:
            json.dump(stats, outfile, indent=4)

    # delete temporary files from the folder
    if delete_temp_files:

        if os.path.isdir(project_folder + "/Tmp-MM-Dir"):
            shutil.rmtree(project_folder + "/Tmp-MM-Dir")
        if os.path.isdir(project_folder + "/Pastis"):
            shutil.rmtree(project_folder + "/Pastis")
        if os.path.isfile(project_folder + "/mm3d-LogFile.txt"):
            os.remove(project_folder + "/mm3d-LogFile.txt")
        for file in os.listdir(project_folder):
            filename = os.fsdecode(file)
            if filename.startswith("MM-Error-") and filename.endswith(".txt"):
                os.remove(project_folder + "/" + filename)


if __name__ == "__main__":
    temp_folder = "/data_1/ATM/data/sfm/projects/filter_test"

    mode = "MulScale"
    img_pat = "OIS*.*tif"
    low_res = 1000
    high_res = 2500
    exp_txt = 1

    args = {
        "Mode": mode,
        "ImagePattern": img_pat,
        "LowResolutionImgSize": low_res,
        "HighResolutionImgSize": high_res,
        "ExpTxt": exp_txt
    }

    tapioca(
        project_folder=temp_folder,
        m_args=args,
        print_output=debug_print,
        print_orig_errors=debug_print_errors
    )
