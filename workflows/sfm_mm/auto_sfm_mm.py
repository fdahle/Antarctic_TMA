import os
import json

import src.sfm_mm.SFMProject as SfmP

PATH_SFM_PROJECTS = "/data/ATM/data_1/sfm/mm_projects"

# 1848_4
# input_name = "1848_4"
# input_ids = ['CA184832V0176', 'CA184832V0177', 'CA184832V0178', 'CA184832V0179',
#              'CA184832V0180', 'CA184832V0181', 'CA184832V0182', 'CA184832V0183']

# 1848_1
input_name = "1848_1_test_new"
input_ids = ['CA184832V0106', 'CA184832V0107', 'CA184832V0108',
             'CA184832V0109', 'CA184832V0110', 'CA184832V0111']

debug = True
overwrite = False
resume = True
georef_mode = "absolute"  # "absolute" or "relative"


def auto_sfm_mm(project_name, image_ids, overwrite=False):

    print("Overwrite:", overwrite)
    print("Resume:", resume)

    # create path to the project
    project_path = os.path.join(PATH_SFM_PROJECTS, project_name)

    # check if the project exists
    if os.path.exists(project_path):
        if resume:
            # we don't need to do anything, but set overwrite to false
            overwrite = False

        else:
            if overwrite is False:  # noqa
                raise FileExistsError(f"Project {project_name} already exists")
            else:
                os.system(f"rm -r {project_path}")

    # init the SFMProject
    sfm_project = SfmP.SFMProject(project_name, PATH_SFM_PROJECTS,
                                  micmac_path="/home/fdahle/micmac/bin/mm3d",
                                  auto_enter=True,
                                  debug=debug, resume=resume, overwrite=overwrite)

    # get the micmac args
    json_path = "project_args.json"
    with open(json_path, 'r') as file:
        micmac_args = json.load(file)

    sfm_project.prepare_files(image_ids, copy_masks=False, create_masks=True, copy_xml=True,
                              create_image_thumbnails=True, create_mask_thumbnails=True,
                              create_camera_positions=True)
    sfm_project.set_camera("1983")

    if georef_mode == "absolute":
        commands = ["ReSampFid", "TapiocaCustom", "Schnaps", "Tapas", "AperiCloud_Relative",
                    "OriConvert", "CenterBascule",
                    "Campari_Absolute", "AperiCloud_Absolute", "Tarama_Absolute",
                    "Malt_Absolute", "Tawny", "Nuage2Ply"]
    elif georef_mode == "relative":
        commands = ["ReSampFid", "ReduceCustom", "TapiocaCustom", "Schnaps", "Tapas",
                    "AperiCloud_Relative", "Campari_Relative", "Tarama_Relative",
                    "Malt_Relative", "Tawny", "Nuage2Ply"]
    else:
        raise ValueError("georef_mode must be 'absolute' or 'relative'")

    sfm_project.start("manual", commands, micmac_args, skip_existing=True,
                      print_all_output=True,
                      save_stats=True, save_raw=True)


if __name__ == "__main__":

    auto_sfm_mm(input_name, input_ids, overwrite=overwrite)
