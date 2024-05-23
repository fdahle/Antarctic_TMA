import os
import json

import src.sfm_mm.SFMProject as SfmP

PATH_SFM_PROJECTS = "/data_1/ATM/data_1/sfm/mm_projects"

input_name = "test2147"
#input_ids = ["CA180132V0094", "CA180132V0095", "CA180132V0096", "CA180132V0097"]

input_ids = ["CA214732V0027", "CA214732V0028", "CA214732V0029", "CA214732V0030", "CA214732V0031", "CA214732V0032", "CA214732V0033", "CA214732V0034","CA214732V0035"]
debug=True
overwrite = False
resume = True
georef_mode = "relative"  # "absolute" or "relative"


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
                                  debug=debug, resume=resume, overwrite=overwrite)

    # get the micmac args
    json_path = "project_args.json"
    with open(json_path, 'r') as file:
        micmac_args = json.load(file)

    sfm_project.prepare_files(image_ids, copy_masks=True, copy_xml=True)
    sfm_project.set_camera("1983")

    if georef_mode == "absolute":
        commands = ["ReSampFid", "TapiocaCustom", "Schnaps", "Tapas", "AperiCloud_Relative",
                    "CenterBascule", "Campari_Absolute", "AperiCloud_Absolute", "Tarama_Absolute",
                    "Malt_Absolute", "Tawny", "Nuage2Ply"]
    elif georef_mode == "relative":
        #commands = ["ReSampFid"]

        import src.sfm_mm.snippets.reduce_images as ri
        ri.reduce_images(os.path.join(PATH_SFM_PROJECTS, project_name), 200)

        commands = ["TapiocaCustom", "Schnaps", "Tapas",
                    "AperiCloud_Relative", "Campari_Relative", "Tarama_Relative",
                    "Malt_Relative", "Tawny", "Nuage2Ply"]
    else:
        raise ValueError("georef_mode must be 'absolute' or 'relative'")

    sfm_project.start("manual", commands, micmac_args, print_all_output=True,
                      save_stats=True, save_raw=True)


if __name__ == "__main__":

    auto_sfm_mm(input_name, input_ids, overwrite=overwrite)
