import glob
import os.path

from src.sfm.mm_commands._base_command import BaseCommand


class ReSampFid(BaseCommand):

    required_args = ["ImagePattern", "ScanResolution"]
    allowed_args = ["ImagePattern", "ScanResolution"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class with all arguments passed to ReSampFid
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

    def build_shell_string(self):

        # build the shell command
        shell_string = f'ReSampFid "{self.mm_args["ImagePattern"]}" ' \
                       f'{self.mm_args["ScanResolution"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        return shell_string

    def validate_mm_parameters(self):

        # adapt the image pattern for glob
        image_pattern = self.mm_args['ImagePattern'].replace("/.*.", "/*.")

        # check if we get images with the image pattern in mm_args
        image_files = glob.glob(self.project_folder + "/" + image_pattern)
        if len(image_files) == 0:
            raise ValueError(f"No images found with pattern {self.mm_args['ImagePattern']}")

        if self.mm_args['ScanResolution'] <= 0:
            raise ValueError("ScanResolution must be greater than 0")

    def validate_required_files(self):

        # check for camera xml files
        if os.path.isfile(self.project_folder + "/MicMac-LocalChantierDescripteur.xml") is False:
            raise FileNotFoundError("MicMac-LocalChantierDescripteur.xml is missing")
        if os.path.isfile(self.project_folder + "/Ori-InterneScan/MeasuresCamera.xml") is False:
            raise FileNotFoundError("MeasuresCamera.xml is missing")

