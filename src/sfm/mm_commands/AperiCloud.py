import os
import glob
import shutil

from src.sfm.mm_commands._base_command import BaseCommand


class AperiCloud(BaseCommand):

    required_args = ["ImagePattern", "Orientation"]
    allowed_args = ["ImagePattern", "Orientation", "ExpTxt", "Out", "Bin",
                    "RGB", "SeuilEc", "LimBsH", "WithPoints", "CalPerIm",
                    "Focs", "WithCam", "ColCadre", "ColRay", "SH"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

    def build_shell_string(self):

        # build the basic shell command
        shell_string = f'AperiCloud {self.mm_args["ImagePattern"]} {self.mm_args["Orientation"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        return shell_string

    def validate_mm_parameters(self):

        if "/" in self.mm_args["ImagePattern"]:
            raise ValueError("ImagePattern cannot contain '/'. Use a pattern like '*.tif' instead.")

    def validate_required_files(self):

        # check the orientation folder
        orientation_folder = self.project_folder + "/Ori-" + self.mm_args["Orientation"]
        if os.path.isdir(orientation_folder) is False:
            raise FileNotFoundError(f"No Orientation folder found at '{orientation_folder}'.")

        # check all tif files in images-subfolder and copy them to the project folder if not already there
        homol_files = glob.glob(self.project_folder + "/images/*.tif")
        for file in homol_files:
            base_name = os.path.basename(file)

            if os.path.isfile(self.project_folder + "/" + base_name) is False:
                shutil.copy(file, self.project_folder + "/" + base_name)

        if "SH" in self.mm_args:
            if os.path.isdir(self.project_folder + "/" + self.mm_args["SH"]) is False:
                raise FileNotFoundError(f"'No SH-folder found at "
                                        f"{self.project_folder + '/' + self.mm_args['SH']}' ")