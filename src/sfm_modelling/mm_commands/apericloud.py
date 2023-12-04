import os
import subprocess
import shutil
import glob
import json

from tqdm import tqdm

debug_print = False
debug_print_errors = True


def apericloud(project_folder, m_args,
               save_stats=True, stats_folder="stats",
               delete_temp_files=True,
               print_output=False, print_orig_errors=False):
    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder

    # set the required arguments
    required_args = ["ImagePattern", "Orientation"]
    allowed_args = ["ImagePattern", "Orientation", "ExpTxt", "Out", "Bin", "RGB", "SeuilEc",
                    "LimBsH", "WithPoints", "CalPerIm", "Focs", "WithCam", "ColCadre", "ColRay", "SH"]

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

        # check required arguments
        orientation_folder = project_folder + "/Ori-" + m_args["Orientation"]
        assert os.path.isdir(orientation_folder), \
            f"'{orientation_folder}' is not a valid path to a folder"

        # check optional arguments
        if "SH" in m_args:
            assert os.path.isdir(project_folder + "/" + m_args["SH"]), \
                f"'{project_folder + '/' + m_args['SH']}' is not a valid path to a folder"

    # validation the input parameters
    __validate_input()

    # here we define the error messages that can happen
    error_dict = {
        "Ori name is not a valid existing directory":
            "blabla"
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {
        "nbUnknown": None,
        "images": {}
    }

    m_args["ImagePattern"] = "OIS.*tif"

    # the actual calling of apericloud
    shell_string = f'mm3d AperiCloud "{m_args["ImagePattern"]}" {m_args["Orientation"]}'

    # add the arguments to the shell string
    for key, val in m_args.items():

        # required arguments are called extra
        if key in required_args:
            continue

        shell_string = shell_string + " " + str(key) + "=" + str(val)

    print("Start with AperiCloud")  # noqa
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
            prefixes = ["NUM", " BEGIN", "END", "NB PACK", '"NKS-Set', "NEW CALIB", "Pack"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part II
            suffixes = ["matches.\n"]
            if stdout_line.endswith(tuple(suffixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part II
            mifixes = ["PastDevlop", "0 0", "occurence of warn type"]  # noqa
            if any(x in stdout_line for x in mifixes):
                if print_output is False:
                    continue

            # skip unnecessary text part IV
            prefixes = ["--- End Iter", "BEGIN", "First", "Com = "]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            if "GUIMBAL" in stdout_line:
                splits = stdout_line.split(" ")
                img = splits[2][10:-4]
                val = splits[3][:-2]
                stats["images"][img] = val
                if print_output is False:
                    continue

            if "APPLI APERO" in stdout_line:
                splits = stdout_line.split(" ")
                stats["nbUnknown"] = splits[-1][:-2]
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

    print("Finished with AperiCloud")

    if save_stats:
        with open(stats_folder + "/apericloud.json", "w") as outfile:
            json.dump(stats, outfile, indent=4)

    # delete temporary files from the folder
    if delete_temp_files:
        if os.path.isdir(project_folder + "/Tmp-MM-Dir"):
            shutil.rmtree(project_folder + "/Tmp-MM-Dir")
        if os.path.isfile(project_folder + "/mm3d-LogFile.txt"):
            os.remove(project_folder + "/mm3d-LogFile.txt")
        if os.path.isfile(project_folder + "/WarnApero.txt"):
            os.remove(project_folder + "/WarnApero.txt")
        for file in os.listdir(project_folder):
            filename = os.fsdecode(file)
            if filename.startswith("MM-Error-") and filename.endswith(".txt"):
                os.remove(project_folder + "/" + filename)
            if filename.startswith("MkStdMM"):
                os.remove(project_folder + "/" + filename)


if __name__ == "__main__":
    temp_folder = "/data_1/ATM/data/sfm/" \
                  "projects/final_test"

    img_pat = "OIS*.*tif"
    orientation = "Relative"
    sh = "Homol_mini"  # "HomolMasqFiltered"
    exp_txt = 1

    args = {
        "ImagePattern": img_pat,
        "Orientation": orientation,
        "SH": sh,
        "ExpTxt": exp_txt
    }

    apericloud(
        project_folder=temp_folder,
        m_args=args,
        print_output=debug_print,
        print_orig_errors=debug_print_errors
    )
