
from src.sfm.mm_commands._base_command import BaseCommand

class Schnaps(BaseCommand):

    required_args = ["ImagePattern"]
    allowed_args = ["ImagePattern", "HomolIn", "NbWin", "ExeWrite", "HomolOut", "ExpTxt",
                    "VeryStrict", "ShowStats", "DoNotFilter", "PoubelleName",
                    "minPercentCoverage", "MoveBadImgs", "OutTrash", "MiniMulti",
                    "NetworkExport"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class with all arguments passed to ReSampFid
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

    def build_shell_string(self):

        # build the basic shell command
        shell_string = f'Schnaps "{self.mm_args["ImagePattern"]}"'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        return shell_string

    def validate_mm_parameters(self):
        pass

    def validate_required_files(self):
        pass

