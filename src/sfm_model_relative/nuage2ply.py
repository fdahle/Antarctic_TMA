import os
import subprocess
import shutil
import json

from tqdm import tqdm

debug_print = False
debug_print_errors = True


def nuage2ply(project_folder, m_args,
              save_stats=True, stats_folder="stats",
              delete_temp_files=True,
              print_output=False, print_orig_errors=False):
    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder

    required_args = ["XmlFile"]
    allowed_args = ["XmlFile", "Sz", "P0", "Out", "Scale", "Attr", "Comments", "Bin", "Mask",
                    "SeuilMask", "Dyn", "DoPly", "DoXYZ", "Normale", "NormByC", "ExagZ",
                    "RatioAttrCarte", "Mesh", "64B", "Offs", "NeighMask", "ForceRGB"]

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

        # check xml file for validity
        assert os.path.isfile(project_folder + "/" + m_args["XmlFile"]), \
            f"'{project_folder}/{m_args['XmlFile']}' is not a valid path to a xml file"

        # check some args for validity
        if "Attr" in m_args:
            assert os.path.isfile(project_folder + "/" + m_args["Attr"]), \
                f"'{project_folder}/{m_args['Attr']}' is not a valid path to a tif file"

    # validation the input parameters
    __validate_input()

    # here we define the error messages that can happen
    error_dict = {
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {
        "nr_of_warnings": 0
    }

    # the actual calling of nuage2ply
    shell_string = f'mm3d Nuage2Ply {m_args["XmlFile"]} '

    # add the arguments to the shell string
    for key, val in m_args.items():

        # required arguments are called extra
        if key in required_args:
            continue

        shell_string = shell_string + " " + str(key) + "=" + str(val)

    print("Start with Nuage2Ply")
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

            # skip unnecessary text part I
            prefixes = ["First context", "First detected"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part II
            suffixes = ["exported\n"]
            if stdout_line.endswith(tuple(suffixes)):
                if print_output is False:
                    continue

            # save nr of warnings
            if "occurence of warn" in stdout_line:
                splits = stdout_line.split(" ")
                stats["nr_of_warnings"] = stats["nr_of_warnings"] + int(splits[0])

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
        with open(stats_folder + "/nuage2ply.json", "w") as outfile:
            json.dump(stats, outfile, indent=4)

    # delete temporary files from the folder
    if delete_temp_files:

        if os.path.isfile(project_folder + "/mm3d-LogFile.txt"):
            os.remove(project_folder + "/mm3d-LogFile.txt")
        for file in os.listdir(project_folder):
            filename = os.fsdecode(file)
            if filename.startswith("MM-Error-") and filename.endswith(".txt"):
                os.remove(project_folder + "/" + filename)


if __name__ == "__main__":
    temp_folder = "/data_1/ATM/data/sfm/projects/func_test"

    xml_file = "MEC-Malt/NuageImProf_STD-MALT_Etape_6.xml"
    attr = "Ortho-MEC-Malt/Orthophotomosaic.tif"
    output_name = "PointCloud.ply"

    args = {
        "XmlFile": xml_file,
        "Attr": attr,
        "Out": output_name
    }

    nuage2ply(
        project_folder=temp_folder,
        m_args=args,
        print_output=debug_print,
        print_orig_errors=debug_print_errors
    )
