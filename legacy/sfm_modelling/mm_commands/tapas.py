import os
import subprocess
import shutil
import glob
import json

from tqdm import tqdm

debug_print = False
debug_print_errors = True

def tapas(project_folder, m_args,
          save_stats=True, stats_folder="stats",
          delete_temp_files=True,
          ignore_warnings = True,
          print_output=False, print_orig_errors=False):

    # get some links to folders and files
    stats_folder = project_folder + "/" + stats_folder

    required_args = ["DistortionModel", "ImagePattern"]
    allowed_args = ["DistortionModel", "ImagePattern", "ExpTxt", "Out", "InCal", "InOri", "DoC",  # noqa
                    "ForCalib", "Focs", "VitesseInit", "PPRel", "Decentre", "PropDiag", "SauvAutom",  # noqa
                    "ImInit", "MOI"]

    additional_args = ["DBF", "Debug", "DegRadMax", "DegGen", "LibAff", "LibDec", "LibPP", "LibCP",
                       "LibFoc", "RapTxt", "LinkPPaPPs", "FrozenPoses", "SH", "RefineAll"]
    additional_args_fraser = ["ImMinMax", "EcMax"]
    additional_args_fraser_basic = ["ImMinMax", "EcMax"]
    additional_args_fish_eye_equi = ["ImMinMax", "EcMax"]
    additional_args_hemi_equi = ["ImMinMax"]

    lst_of_distortion_models = ["RadialBasic", "RadialStd", "RadialExtended", "FraserBasic",
                                "Fraser", "FishEyeEqui", "FE_EquiSolBasic", "FishEyeBasic",
                                "FishEyeStereo", "Four", "AddFour", "AddPolyDeg", "Ebner",  # noqa
                                "Brown", "AutoCal", "Figee", "HemiEqui"]  # noqa

    # extend allowed arguments based on distortion model
    if m_args["DistortionModel"] == "RadialBasic":
        allowed_args = allowed_args + additional_args
    elif m_args["DistortionModel"] == "RadialExtended":
        allowed_args = allowed_args + additional_args
    elif m_args["DistortionModel"] == "Fraser":
        allowed_args = allowed_args + additional_args + additional_args_fraser
    elif m_args["DistortionModel"] == "FraserBasic":
        allowed_args = allowed_args + additional_args + additional_args_fraser_basic
    elif m_args["DistortionModel"] == "FishEyeEqui":
        allowed_args = allowed_args + additional_args + additional_args_fish_eye_equi
    elif m_args["DistortionModel"] == "HemiEqui":
        allowed_args = allowed_args + additional_args + additional_args_hemi_equi

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

        # check Distortion Model for validity
        assert m_args["DistortionModel"] in lst_of_distortion_models, \
            f"'{m_args['DistortionModel']}' is not a valid calibration model"

        # check if only allowed arguments were used
        for arg in m_args:
            assert arg in allowed_args, f"{arg} is not an allowed argument"

        # check some args for validity
        if "SH" in m_args:
            assert os.path.isdir(project_folder + "/" + m_args["SH"]), \
            f"'{project_folder + '/' + m_args['SH']}' is not a valid path to a folder"
        if "InCal" in m_args:
            assert os.path.isdir(project_folder + "/" + m_args["InCal"]), \
                f"'{project_folder + '/' + m_args['InCal']}' is not a valid path to a folder"

        # get the images we are working with
        _lst_images = []
        for _elem_ in glob.glob(project_folder + "/" + m_args["ImagePattern"]):
            _img_name = _elem_.split("/")[-1].split(".")[0]
            _lst_images.append(_img_name)

        # check if we can find images
        assert len(_lst_images) > 0, "No images could be found with this image pattern"

    # validate the input parameters
    __validate_input()

    # define some values
    last_id = None

    # here we define the error messages that can happen
    error_dict = {
        "Not Enouh Equation in ElSeg3D::L2InterFaisceaux":  # noqa
            "Lack of tie points that are usable",
        "cPackObsLiaison::ObsMulOfName":
            f"No tie-points could be found for '{last_id}'",
        "aBestCam==0":
            "Not enough tie points for the images (no overlap or no common features)",
        "Distortion Inversion  by finite difference do not converge":
            f"Geometry of object not good enough, need more tie points",
        "very singular matrix in Gausj": "TODO"  # noqa
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {
        "camera": "",
        "focal_length": "",
        "nb_unknown": "",
        "images": {},
        "points": {
            "Ok": [],
            "Ok_perc": [],
            "InsufPoseInit": [],
            "InsufPoseInit_perc": [],
            "PdsResNull": [],
            "PdsResNull_perc": [],
            "VisibIm": [],
            "VisibIm_perc": [],
        },
        "point_stats": {
            "residual": [],
            "res_moy": [],
            "res_max": [],
            "worst_res": [],
            "worst_res_img": [],
            "worst_res_perc": [],
            "worst_res_perc_img": [],
            "cond_avg": [],
            "cond_max": []
        }
    }

    # create the basic shell string
    shell_string = f'mm3d Tapas {m_args["DistortionModel"]} "{m_args["ImagePattern"]}"'

    # add the arguments to the shell string
    for key, val in m_args.items():

        # required arguments are called extra
        if key in required_args:
            continue

        shell_string = shell_string + " " + str(key) + "=" + str(val)

    print("Start with Tapas")
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

            # ignore warnings
            if ignore_warnings:

                if stdout_line.startswith(" Warn tape enter"):
                    #p.communicate(input="\n")
                    pass

                if stdout_line.startswith("ValueNeg"):
                    if print_output is False:
                        continue

            # skip unnecessary text part I
            suffixes = ["matches.\n"]
            if stdout_line.endswith(tuple(suffixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part II
            prefixes = ["BEGIN Pre-compile", "BEGIN Load Observation", "BEGIN Init Inconnues",
                        "BEGIN Compensation", "BEGIN AMD", "END AMD", "NB PACK", "Pack Obs"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part II
            prefixes = ["xProf", "LIB ", "Lib PP",  "--- End Iter", "----- Stat on"]
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part III
            prefixes = ["NO GUIMBAL", "  Add Pose", "Com = ", " MdPppppF", "---- PROFS", "NUM", "MST"]  # noqa
            if stdout_line.startswith(tuple(prefixes)):
                if print_output is False:
                    continue

            # skip unnecessary text part IV
            mifixes = ["; Pere :"]
            if any(x in stdout_line for x in mifixes):
                if print_output is False:
                    continue

            # get the focal length and camera
            if stdout_line.startswith("NEW CALIB"):
                f_pos = stdout_line.find("Foc-") + 4
                c_pos = stdout_line.find("_Cam-") + 5

                focal_length = stdout_line[f_pos:c_pos-5]
                camera = stdout_line[c_pos:]
                camera = camera.rstrip("\n")

                stats["camera"] = camera
                stats["focal_length"] = focal_length

                if print_output is False:
                    continue

            # get the number of unknown
            if stdout_line.startswith("APPLI APERO"):
                splits = stdout_line.split(" = ")
                stats["nb_unknown"] = splits[1].rstrip("\n")
                if print_output is False:
                    continue

            # get the results per image
            if stdout_line.startswith("RES:"):

                # split string and get image image_id
                splits = stdout_line.split(" ")
                img = splits[0][15:-8]

                # check if image already in dict and if not create new dict
                if img not in stats["images"]:
                    stats["images"][img] = {
                        "total_res": [],
                        "percentage_invalid": [],
                        "tie_points_invalid": [],
                        "tie_points_total": [],
                        "tie_points_multiple": [],
                        "tie_points_multiple_good": [],
                        "time":[]
                    }

                # append the data
                stats["images"][img]["total_res"].append(float(splits[2]))
                percentage_under_ecart_max = splits[4]
                percentage_under_ecart_max = round(100 - float(percentage_under_ecart_max), 3)
                stats["images"][img]["percentage_invalid"].append(percentage_under_ecart_max)
                total_num_tie_points = int(splits[6])
                tie_points_invalid = int(total_num_tie_points * percentage_under_ecart_max/100)
                stats["images"][img]["tie_points_invalid"].append(tie_points_invalid)
                stats["images"][img]["tie_points_total"].append(total_num_tie_points)
                stats["images"][img]["tie_points_multiple"].append(int(splits[8]))
                stats["images"][img]["tie_points_multiple_good"].append(int(splits[10]))
                stats["images"][img]["time"].append(splits[12].strip())

                if print_output is False:
                    continue

            # get the stats for all points
            if stdout_line.startswith("     *   Perc"):
                splits = stdout_line.split(" ")
                if splits[-1].strip() == "Ok":
                    stats["points"]["Ok"].append(splits[11][3:])
                    stats["points"]["Ok_perc"].append(splits[8][5:-1])
                if splits[-1].strip() == "InsufPoseInit":
                    stats["points"]["InsufPoseInit"].append(splits[11][3:])
                    stats["points"]["InsufPoseInit_perc"].append(splits[8][5:-1])
                if splits[-1].strip() == "PdsResNull":
                    stats["points"]["PdsResNull"].append(splits[11][3:])
                    stats["points"]["PdsResNull_perc"].append(splits[8][5:-1])
                if splits[-1].strip() == "BSurH":
                    stats["points"]["BSurH"].append(splits[11][3:])
                    stats["points"]["BSurH_perc"].append(splits[8][5:-1])
                if splits[-1].strip() == "VisibIm":
                    stats["points"]["VisibIm"].append(splits[11][3:])
                    stats["points"]["VisibIm_perc"].append(splits[8][5:-1])

                if print_output is False:
                    continue

            if stdout_line.startswith("| |  "):

                # split the line to get the different values
                splits = stdout_line.split(" ")

                if stdout_line.startswith("| |  Residual"):
                    stats["point_stats"]["residual"].append(float(splits[5].strip()))
                    if len(splits) > 6:
                        stats["point_stats"]["res_moy"].append(float(splits[8][4:]))
                        stats["point_stats"]["res_max"].append(float(splits[9][5:].strip()))
                    else:
                        stats["point_stats"]["res_moy"].append('')
                        stats["point_stats"]["res_max"].append('')

                elif stdout_line.startswith("| |  Worst"):
                    stats["point_stats"]["worst_res"].append(float(splits[5]))
                    stats["point_stats"]["worst_res_img"].append(splits[7])
                    stats["point_stats"]["worst_res_perc"].append(float(splits[10]))
                    stats["point_stats"]["worst_res_perc_img"].append(splits[12].strip())
                elif stdout_line.startswith("| |  Cond"):
                    stats["point_stats"]["cond_avg"].append(float(splits[6]))
                    stats["point_stats"]["cond_max"].append(float(splits[8]))

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

    print("Finished with Tapas")

    if save_stats:
        with open(stats_folder + "/tapas.json", "w") as outfile:
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

    dist_model = "RadialBasic"
    img_pat = "OIS*.*tif"
    sh = "Homol_mini"  # "HomolMasqFiltered"
    out = "Relative"
    lib_foc = 0
    exp_txt = 1

    args = {
        "DistortionModel": dist_model,
        "ImagePattern": img_pat,
        "Out": out,
        "SH": sh,
        "LibFoc": lib_foc,
        "ExpTxt": exp_txt
    }

    tapas(
        project_folder=temp_folder,
        m_args=args,
        print_output=debug_print,
        print_orig_errors=debug_print_errors
    )
