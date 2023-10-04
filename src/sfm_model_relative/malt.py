import os
import subprocess
import shutil
import glob
import json

from tqdm import tqdm

debug_print = False
debug_print_errors = True

def malt(project_folder, m_args,
         save_stats=True, stats_folder="stats",
         delete_temp_files=True,
         print_output=False, print_orig_errors=False):

    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder

    required_args = ["Mode", "ImagePattern", "Orientation"]
    allowed_args = ["Mode", "ImagePattern", "Orientation", "Master", "SzW", "CorMS", "UseGpu", "Regul",
                    "DirMEC", "DirOF", "UseTA", "ZoomF", "ZoomI", "ZPas", "Exe", "Repere", "NbVI", "HrOr",
                    "LrOr", "DirTA", "Purge", "DoMEC", "DoOrtho", "UnAnam", "2Ortho", "ZInc", "DefCor",
                    "CostTrans", "Etape0", "AffineLast", "ResolOrtho", "ImMNT", "ImOrtho", "ZMoy",
                    "Spherik", "WMI", "vMasqIm", "MasqImGlob", "IncMax", "BoxClip", "BoxTerrain", "ResolTerrain",
                    "RoundResol", "GCC", "EZA", "Equiv", "MOri", "MaxFlow", "SzRec", "Masq3D",
                    "NbProc", "PSIBN", "InternalNoIncid", "PtDebug"]

    lst_of_modes = ["Ortho", "UrbanMNE", "GeomImage"]

    # function to check if the input parameters are valid
    def __validate_input():

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

        # check some args for validity
        assert m_args["Mode"] in lst_of_modes, \
            f"'{m_args['Mode']}' is not a valid mode"

        # get the images we are working with
        _lst_images = []
        for _elem_ in glob.glob(project_folder + "/" + m_args["ImagePattern"]):
            _img_name = _elem_.split("/")[-1].split(".")[0]
            _lst_images.append(_img_name)

        # check if we can find images
        assert len(_lst_images) > 0, "No images could be found with this image pattern"

    __validate_input()

    # here we define the error messages that can happen
    error_dict = {
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {
        "params":{
        },
        "TCor": [],
        "CTimeC":[],
        "TOpt": [],
        "R2": [],
        "RN": [],
        "R-GEN": [],
        "Isol": [],
        "PT": []
    }

    # the actual calling of malt
    shell_string = f'mm3d Malt {m_args["Mode"]} "{m_args["ImagePattern"]}" {m_args["Orientation"]}'

    # add the arguments to the shell string
    for key, val in m_args.items():

        # required arguments are called extra
        if key in required_args:
            continue

        shell_string = shell_string + " " + str(key) + "=" + str(val)

    print("Start with Malt")
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
            suffixes = ["matches.\n", "Loaded\n", "Begin Opt\n"]
            if stdout_line.endswith(tuple(suffixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part II
            prefixes = ["<< Make Pyram", "PDV--Make Masq", ">>  Done Masq", "<< Make Masque", "TA : ",
                        "aSzClip", "make:", "cp ", " ---End", " ---Launch"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part III
            mifixes = ["Reduc2MM", "BEGIN BLOC", "/micmac/bin/mm3d", "tifDeZoom"]
            if any(x in stdout_line for x in mifixes):
                if print_output is False:
                    continue

            # skip unnecessary text part IV
            prefixes = ["DO ONE BLOC", "-------- BEGIN", ">> Done Masq", "<< Make Masque", "TA : ",
                        "aSzClip", "==============================", "PC-Name",
                        "============= PARAMS"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # get params
            if stdout_line.startswith(" -  SzWindow"):
                stats["params"]["SzWindow"] = stdout_line.split(" ")[4]
                if print_output is False:
                    continue

            if stdout_line.startswith(" -  Regul"):
                stats["params"]["Regul"] = stdout_line.split(" ")[4]
                if print_output is False:
                    continue

            if stdout_line.startswith(" -  Final Zoom"):
                stats["params"]["FinalZoom"] = stdout_line.split(" ")[4]
                if print_output is False:
                    continue

            if stdout_line.startswith(" -  Initial Zoom"):
                stats["params"]["InitialZoom"] = stdout_line.split(" ")[4]
                if print_output is False:
                    continue

            if stdout_line.startswith(" -  Use TA as Mask"):
                stats["params"]["TAasMask"] = stdout_line.split(" ")[4]
                if print_output is False:
                    continue

            if stdout_line.startswith(" -  Z Step"):
                stats["params"]["ZStep"] = stdout_line.split(" ")[4]
                if print_output is False:
                    continue

            if stdout_line.startswith(" -  Nb Min Visible Images"):
                stats["params"]["NrMinVisibleImages"] = stdout_line.split(" ")[4]
                if print_output is False:
                    continue

            # get some statistics
            if stdout_line.startswith("       TCor"):
                splits = stdout_line.lstrip().split(" ")

                stats["TCor"].append(splits[1])
                stats["CTimeC"].append(splits[3])
                stats["TOpt"].append(splits[5])
                stats["R2"].append(splits[9])
                stats["RN"].append(splits[11])
                stats["R-GEN"].append(splits[15])
                stats["Isol"].append(splits[17])
                stats["PT"].append(splits[21][:-2])

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


    print("Finished with Malt")

    if save_stats:
        with open(stats_folder + "/malt.json", "w") as outfile:
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

    mode = "Ortho"
    image_pattern = "OIS*.*tif"
    orientation = "Relative"
    nbvi = 2
    zoom_f = 8
    def_cor = 0
    cost_trans = 1
    eza = 1  # use metric units

    args = {
        "Mode": mode,
        "ImagePattern": image_pattern,
        "Orientation": orientation,
        "NbVI": nbvi,
        "ZoomF": zoom_f,
        "DefCor": def_cor,
        "CostTrans": cost_trans,
        "EZA": eza
    }

    malt(
        project_folder=temp_folder,
        m_args=args,
        print_output=debug_print,
        print_orig_errors=debug_print_errors
    )