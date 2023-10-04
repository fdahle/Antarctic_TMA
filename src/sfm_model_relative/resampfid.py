import os
import subprocess
import shutil
import glob
import json
import xml.etree.ElementTree as ET

from xml.etree.ElementTree import parse
from tqdm import tqdm


# __DONE __ #

debug_print = False
debug_print_errors = True

def resampfid(project_folder, m_args,
              resample_masks=True, move_images=True,
              save_stats=True, stats_folder="stats",
              delete_temp_files=True,
              print_output=False, print_orig_errors=False):

    # get some links to folders and files
    xml_folder = project_folder + "/" + "Ori-InterneScan"
    stats_folder = project_folder + "/" + stats_folder

    # set the required arguments
    required_args = ["ImagePattern", "ScanResolution"]

    # function to check if the input parameters are valid
    def __validate_input():

        # check required folders
        assert os.path.isdir(project_folder), \
            f"'{project_folder}' is not a valid path to a folder"
        assert os.path.isdir(xml_folder), \
            f"'Ori-InterneScan' is missing in '{project_folder}'"
        if save_stats:
            assert os.path.isdir(stats_folder), \
                f"stats folder is missing at '{stats_folder}'"

        # check if we have the required arguments
        for r_arg in required_args:
            assert r_arg in m_args, f"{r_arg} is a required argument"

        assert m_args["ScanResolution"] > 0, \
            "Scan resolution must be bigger than 0"

        # check required files
        assert os.path.isfile(project_folder + "/MicMac-LocalChantierDescripteur.xml"), \
            f"LocalChantierDescripteur.xml is missing"
        assert os.path.isfile(xml_folder + "/MeasuresCamera.xml"), \
            f"MeasuresCamera.xml is missing"

        # get the images we are working with
        _lst_images = []
        _lst_masks = []
        for _elem_ in glob.glob(project_folder + "/" + m_args["ImagePattern"]):
            img_name = _elem_.split("/")[-1].split(".")[0]
            if img_name.startswith("OIS-Reech_"):
                continue

            if img_name.endswith("_mask"):
                _lst_masks.append(img_name)
            else:
                _lst_images.append(img_name)

        # check if we can find images
        assert len(_lst_images) > 0, "No images could be found with this image pattern"

        # assert we have for each image the suitable xml file and the suitable mask
        for _elem_ in _lst_images:
            assert os.path.isfile(xml_folder + "/MeasuresIm-" + _elem_ + ".tif.xml"), \
                f"No xml file for {_elem_}"
            if resample_masks:
                assert _elem_ + "_mask" in _lst_masks, \
                    f"No mask file for {_elem_}"

        return _lst_images, _lst_masks

    # validate the input and get at the same time a list of images and masks
    lst_images, lst_masks = __validate_input()

    # copy the xml files for the masks
    if resample_masks:
        for elem in lst_images:
            # copy the xml file
            src = xml_folder + "/MeasuresIm-" + elem + ".tif.xml"
            dst = xml_folder + "/MeasuresIm-" + elem + "_mask.tif.xml"
            shutil.copy(src, dst)

            tree = ET.parse(dst).getroot()
            xmlstr = ET.tostring(tree, encoding='unicode')
            xmlstr = xmlstr.replace(elem, elem + "_mask")
            tree = ET.ElementTree(ET.fromstring(xmlstr))
            tree.write(dst, encoding="UTF-8", xml_declaration=True)

    # calculate boxCh
    tree = parse(project_folder + "/MicMac-LocalChantierDescripteur.xml")
    root = tree.getroot()
    el = root.find("ChantierDescripteur/LocCamDataBase/CameraEntry/SzCaptMm").text
    max_x = float(el.split(" ")[0])
    max_y = float(el.split(" ")[1])
    box_ch = [0, 0, max_x, max_y]

    # here we define the error messages that can happen
    error_dict = {
        "Unexpected End Of String in ElStdRead(vector<Type> &)":
            "Unnecessary spaces in boxCh",
    }
    error_msg = "An undefined error has happened"

    # in this dict we save the stats
    stats = {
        "residuals": {},
        "epip": {}
    }

    # adapt box_ch
    box_ch = str(box_ch)
    box_ch = box_ch.replace(" ", "")

    # add "." to ImagePattern
    m_args["ImagePattern"] = "." + m_args["ImagePattern"]

    # create the shell string
    shell_string = f'mm3d ReSampFid "{m_args["ImagePattern"]}" ' \
                   f'{m_args["ScanResolution"]} BoxCh={box_ch}'

    print("Start with ResampFid")  # noqa
    print(shell_string)

    with subprocess.Popen(["/bin/bash", "-i", "-c", shell_string],
                          cwd=project_folder,
                          stdout=subprocess.PIPE,
                          universal_newlines=True
                          ) as p:

        line_counter = 0  # serves as a progress bar

        for stdout_line in tqdm(p.stdout):

            line_counter += 1

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

            if stdout_line.endswith(" matches.\n"):
                if print_output is False:
                    continue

            # check for the residuals
            if "RESIDU" in stdout_line:  # noqa
                img = stdout_line.split(" ")[1][:-4]
                residual = stdout_line.split(" ")[3]
                stats["residuals"][img] = residual
                if print_output is False:
                    continue

            if "=== RESAMPLE EPIP" in stdout_line:
                img = stdout_line.split(" ")[3][:-4]
                ker = stdout_line.split(" ")[4][4:]
                step = stdout_line.split(" ")[5][5:]
                sz_red = stdout_line.split(" ")[6][6:-7]
                stats["epip"][img] = {
                    "ker": ker,
                    "step": step,
                    "sz_red": sz_red
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

    print("Finished with ResampFid")  # noqa

    if move_images:
        new_folder_path = project_folder + "/images_original"
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        for _elem in lst_images:
            shutil.move(project_folder + "/" + _elem + ".tif",
                        new_folder_path + "/" + _elem + ".tif")

        if resample_masks:

            # copy the original mask files
            new_folder_path = project_folder + "/masks_original"
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            for _elem in lst_masks:
                shutil.move(project_folder + "/" + _elem + ".tif",
                            new_folder_path + "/" + _elem + ".tif")

            # copy the resampled mask files
            new_folder_path = project_folder + "/masks_resampled"
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            for _elem in lst_images:
                shutil.move(project_folder + "/OIS-Reech_" + _elem + "_mask.tif",
                            new_folder_path + "/mask_OIS-Reech_" + _elem + ".tif")

    if save_stats:
        with open(stats_folder + "/resampfid.json", "w") as outfile:
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

    img_pat = "*.tif"
    scan_res = 0.025

    args = {
        "ImagePattern": img_pat,
        "ScanResolution": scan_res
    }

    resampfid(
        project_folder=temp_folder,
        m_args=args,
        print_output=debug_print,
        print_orig_errors=debug_print_errors
    )
