import os
import subprocess
import shutil
import glob
import json

from bs4 import BeautifulSoup as bs4
from tqdm import tqdm

import filter_tie_points as ftp

debug_print = False
debug_print_errors = True


def homolfiltermasq(project_folder, m_args,
                    mask_folder = "masks_resampled", mask_pattern="mask_OIS*.*tif",
                    save_stats=True, stats_folder="stats",
                    delete_temp_files=True,
                    print_output = False, print_orig_errors = False,
                    use_alternative_method=False):

    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder
    mask_folder = project_folder + "/" + mask_folder

    required_args = ["ImagePattern"]
    allowed_args = ["ImagePattern", "PostPlan", "GlobalMasq", "KeyCalculMasq", "ReyEquivNoMasq",
                    "Resol", "ANM", "ExpTxt", "PostIn", "PostOut", "OriMasq3D", "Masq3D",
                    "SelecTer", "DistId", "DistH"]

    # function to check if the input parameters are valid
    def __validate_input():

        # check required folders
        assert os.path.isdir(project_folder), \
            f"'{project_folder}' is not a valid path to a folder"
        assert os.path.isdir(mask_folder), \
            f"'{mask_folder}' is not a valid path to a folder"
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
            _lst_images.append(_elem_.split("/")[-1].split(".")[0])

        _lst_masks = []
        for _elem_ in glob.glob(mask_folder + "/" + mask_pattern):
            _lst_masks.append(_elem_.split("/")[-1].split(".")[0])

        # check if we can find images
        assert len(_lst_images) > 0, "No images could be found with this image pattern"
        assert len(_lst_images) == len(_lst_masks), \
            f"The number of masks ({len(_lst_masks)}) and images ({len(_lst_images)}) is unequal"

    __validate_input()

    if use_alternative_method:

        assert "ExpTxt" in m_args and m_args["ExpTxt"] == 1, \
            "The alternative method only works with txt-files"

        print("Start with alternative ResampFid")  # noqa

        # get homol folder
        homol_folder = project_folder + "/Homol"

        if "KeyCalculMasq" in m_args:
            with open(project_folder + "/MicMac-LocalChantierDescripteur.xml" , 'r') as f:
                data = f.read()
                bs = bs4(data, "lxml")
                key_elems = bs.find_all('keyednamesassociations')
                for elem in key_elems:
                    if elem.find('key').getText() != m_args["KeyCalculMasq"]:
                        continue

                    mask_val = elem.find('calcname').getText()
                    mask_path = project_folder + "/" + mask_val
        elif "GlobalMasq" in m_args:
            mask_path = project_folder + "/" + m_args["GlobalMasq"]

        # iterate all files
        for path, sub_dirs, files in os.walk(homol_folder):
            for name in files:

                if name.endswith("txt") is False:
                    continue

                file_path = os.path.join(path, name)

                mask_1 = file_path.split("/")[-2][6:-4]
                mask_2 = file_path.split("/")[-1][:-8]

                if "$0" in mask_path:
                    mask_1_path = mask_path.replace("$0", mask_1) + ".tif"
                    mask_2_path = mask_path.replace("$0", mask_2) + ".tif"
                else:
                    mask_1_path = mask_path
                    mask_2_path = mask_path

                ftp.filter_tie_points(file_path, mask_1_path, mask_2_path)

        print("Finished with HomolFilterMasq")  # noqa

        return

    # fill some keys in m_args so that the error dict is working
    if "KeyCalculMasq" not in m_args:
        m_args["KeyCalculMasq"] = ""

    # here we define the error messages that can happen
    error_dict = {
        "Cannot get keyed association":
            f"The key '{m_args['KeyCalculMasq']}' for 'KeyCalculMasq' is not in 'LocalChantierDescripteur.xml'",
        "Cannot compute association":
            f"Cannot find the mask value"
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {
        "pairs": {}
    }

    # create the  shell string
    shell_string = f'mm3d HomolFilterMasq "{m_args["ImagePattern"]}"'  # noqa

    # add the arguments to the shell string
    for key, val in m_args.items():

        # required arguments are called extra
        if key in required_args:
            continue

        # only add keys with content
        if len(str(m_args[key])) == 0:
            continue

        shell_string = shell_string + " " + str(key) + "=" + str(val)

    print("Start with ResampFid")  # noqa
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
            prefixes = ["Prep Reste ", "Filter Reste", "For "]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part III
            mifixes = ["["]
            if any(x in stdout_line for x in mifixes):
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

    print("Finished with HomolFilterMasq")  # noqa

    if save_stats:
        with open(stats_folder + "/homolfiltermasq.json", "w") as outfile:
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

    temp_folder = "/data_1/ATM/data/sfm/projects/" \
                  "final_test"

    img_pat = "OIS*.*tif"
    key_calcul_masq = "MyKeyCalculMasq"
    exp_txt = 1

    args = {
        "ImagePattern": img_pat,
        "KeyCalculMasq": key_calcul_masq,
        #"GlobalMasq": "mask.tif",
        "ExpTxt": exp_txt
    }

    homolfiltermasq(
        project_folder=temp_folder,
        m_args=args,
        print_output=debug_print,
        print_orig_errors=debug_print_errors,
        use_alternative_method=True
    )
