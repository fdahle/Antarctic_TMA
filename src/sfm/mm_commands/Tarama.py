# Custom imports
from src.sfm.mm_commands._base_command import BaseCommand


class Tarama(BaseCommand):

    required_args = ["ImagePattern", "Orientation"]
    allowed_args = ["ImagePattern", "Orientation", "Zoom", "Repere", "Out", "ZMoy", "KNadir",
                    "IncMax", "UnUseAXC"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the mm_args
        self.validate_mm_args()

        # validate the input parameters
        self.validate_mm_parameters()

    def build_shell_dict(self):

        shell_dict = {}

        # build the basic shell command
        shell_string = f'Tarama {self.mm_args["ImagePattern"]} {self.mm_args["Orientation"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["Tarama"] = shell_string

        return shell_dict

    def extract_stats(self, raw_output):
        pass

    def validate_mm_parameters(self):

        if "/" in self.mm_args["ImagePattern"]:
            raise ValueError("ImagePattern cannot contain '/'. Use a pattern like '*.tif' instead.")

    def validate_required_files(self):
        pass
