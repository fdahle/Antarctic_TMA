import os.path
import shutil
import subprocess

"""Ideas:
- have a steps.txt file in the project folder, which contains the steps that were done
- create html file with shows stuff
"""


class SFMProject(object):

    # list of supported commands
    valid_commands = [
        "AperiCloud",
        "Campari",
        "GCPBascule"
        "GCPConvert",
        "HomolFilterMasq",
        "Malt",
        "Nuage2Ply",
        "Schnaps",
        "ReSampFid",
        "Tapas",
        "Tapioca",
        "Tarama",
        "Tawny"]

    def __init__(self, project_name, fld,
                 resume=True, overwrite=False):

        # project settings
        self.project_name = project_name
        self.project_folder = fld + "/" + project_name

        # check project settings
        if len(project_name) == 0:
            raise Exception("Project name cannot be empty")
        if os.path.isdir(fld) is False:
            raise Exception("Path to project folder is invalid")

        # check resume and overwrite
        if resume and overwrite:
            raise Exception("Cannot resume and overwrite at the same time")

        # check if the project folder exists
        if os.path.isdir(self.project_folder):
            if resume:
                # do nothing
                pass
            elif not overwrite:
                raise Exception("Project folder already exists")
            else:
                # remove folder and all its content
                shutil.rmtree(self.project_folder)

        # otherwise create the project folder and create the project structure
        #os.mkdir(self.project_folder)
        #self._create_project_structure()

    def set_camera(self):
        pass

    def set_images(self, image_ids):
        self.image_ids = image_ids

    def start(self):

        # check if there are enough images
        if len(self.image_ids < 3):
            raise Exception("Need at least 3 images for SfM")

        # check if every image has an image xml

        # check if there's a camera xml

        pass


    def _copy_files(self):
        pass

    def _create_project_structure(self):

        # create folder for homol
        if os.path.isdir(self.project_folder + "/Homol") is False:
            os.mkdir(self.project_folder + "/Homol")

        # create folder for the xml files
        if os.path.isdir(self.project_folder + "/Ori-InterneScan") is False:
            os.mkdir(self.project_folder + "/Ori-InterneScan")

        # create folder for images resampled
        if os.path.isdir(self.project_folder + "/images_orig") is False:
            os.mkdir(self.project_folder + "/images_orig")

        # create folder for images
        if os.path.isdir(self.project_folder + "/images") is False:
            os.mkdir(self.project_folder + "/images")

        # create folder for masks
        if os.path.isdir(self.project_folder + "/masks_orig") is False:
            os.mkdir(self.project_folder + "/masks_orig")

        # create folder for resampled masks
        if os.path.isdir(self.project_folder + "/masks") is False:
            os.mkdir(self.project_folder + "/masks")

        # create folder for the stats
        if os.path.isdir(self.project_folder + "/stats") is False:
            os.mkdir(self.project_folder + "/stats")

        # create folder for Ori-Relative
        if os.path.isdir(self.project_folder + "/Ori-Relative") is False:
            os.mkdir(self.project_folder + "/Ori-Relative")

        # create folder for Ori-Tapas
        if os.path.isdir(self.project_folder + "/Ori-Tapas") is False:
            os.mkdir(self.project_folder + "/Ori-Tapas")

    def _execute_command(self):
        pass
