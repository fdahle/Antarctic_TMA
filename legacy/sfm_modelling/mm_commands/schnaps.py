import os
import shutil
import subprocess
import glob
import json

from tqdm import tqdm

debug_print = True
debug_print_errors = True

"""
SCHNAPS:
filter tie-points
"""


def schnaps(project_folder, m_args,
            save_stats=True, stats_folder="stats",
            delete_temp_files=True,
            print_output=False, print_orig_errors=False):

    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder

    required_args = ["ImagePattern"]
    allowed_args = ["ImagePattern", "HomolIn", "NbWin", "ExeWrite", "HomolOut", "ExpTxt",
                    "VeryStrict", "ShowStats", "DoNotFilter", "PoubelleName", "minPercentCoverage",
                    "MoveBadImgs", "OutTrash", "MiniMulti", "NetworkExport"]

    # function to check if the input parameters are valid
    def __validate_input():

        # check required folders
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

        # check if we can find images
        assert len(_lst_images) > 0, "No images could be found with this image pattern"

        return _lst_images

    lst_images = __validate_input()

    # here we define the error messages that can happen
    error_dict = {
        "LArgMain , Don't understand": "An input variable is wrong.",
        "No Homol file found": "The files in Homol are probably empty"
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {"images_rejected": 0, "images": {}}
    for img_name in lst_images:
        stats["images"][img_name] = {}

    # the actual calling of schnaps
    shell_string = f'mm3d Schnaps "{m_args["ImagePattern"]}"'

    # add the arguments to the shell string
    for key, val in m_args.items():

        # required arguments are called extra
        if key in required_args:
            continue

        shell_string = shell_string + " " + str(key) + "=" + str(val)

    print("Start with Schnaps")
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

            # skip unnecessary text part II
            prefixes = ["Schnaps : ", "S t", "C h", "H o", "N e", "A c", "P a", "S p", "Bye"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part III
            prefixes = ["Found ", "All sizes:", "Read packs", "Create new homol", "Write new packs",
                        "You can look at", "Quit", "Working dir", "Images pattern", "All sizes", "  * ",
                        "Write new Packs", "Number of searching"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part III
            suffixes = ["matches.\n"]
            if stdout_line.endswith(tuple(suffixes)):
                if print_output is False:
                    continue

            # get stats
            if stdout_line.startswith(" Picture"):

                splits = stdout_line.split(" ")
                img_name = splits[2][:-1].split(".")[0]

                # ignore no homol files
                if ("No homol file") in stdout_line:
                    print(f"NO HOMOL FOR {img_name}")
                    continue
                else:
                    stats["images"][img_name]["files"] = splits[4]
                    stats["images"][img_name]["couples"] = splits[8]
                    if print_output is False:
                        continue

            # get stats part II
            if stdout_line.startswith(" -"):
                splits = stdout_line.split(" ")
                img_name = splits[2][:-1].split(".")[0]
                stats["images"][img_name]["percentage_covered"] = splits[3][:-1]
                stats["images"][img_name]["points_covered"] = splits[8][1:]
                if print_output is False:
                    continue

            # get stats part III
            if stdout_line.endswith("rejected.\n"):
                splits = stdout_line.split(" ")
                num_rejected = splits[0]
                stats["images_rejected"] = num_rejected
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
            if stdout_line.startswith("Bye  (press enter)") or \
                    "An undefined error has happened" in stdout_line:

                print(ValueError(error_msg))

                p.kill()
                exit()

            # the last resort: everything we didn't catch before is printed here
            print(stdout_line, end="")

    print("Finished with Schnaps")  # noqa

    if save_stats:
        with open(stats_folder + "/schnaps.json", "w") as outfile:
            json.dump(stats, outfile, indent=4)

    # delete temporary files from the folder
    if delete_temp_files:
        if os.path.isdir(project_folder + "/Tmp-MM-Dir"):
            shutil.rmtree(project_folder + "/Tmp-MM-Dir")
        if os.path.isfile(project_folder + "/mm3d-LogFile.txt"):
            os.remove(project_folder + "/mm3d-LogFile.txt")
        for file in os.listdir(project_folder):
            filename = os.fsdecode(file)
            if filename.startswith("MM-Error-") and filename.endswith(".txt"):
                os.remove(project_folder + "/" + filename)


if __name__ == "__main__":
    temp_folder = "/data_1/ATM/data/sfm/" \
                  "projects/final_test"

    img_pat = "OIS*.*tif"
    homol_in = "MasqFiltered"
    exp_txt = 1

    args = {
        "ImagePattern": img_pat,
        "HomolIn": homol_in,
        "ExpTxt": exp_txt
    }

    schnaps(
        project_folder=temp_folder,
        m_args=args,
        print_output=debug_print,
        print_orig_errors=debug_print_errors
    )