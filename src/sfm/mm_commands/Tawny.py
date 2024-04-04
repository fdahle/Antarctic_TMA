# Custom imports
from src.sfm.mm_commands._base_command import BaseCommand


class Tawny(BaseCommand):

    required_args = ["DataDirectory"]
    allowed_args = ["DataDirectory", "RadiomEgal", "DEq", "DEqXY", "AddCste", "DegRap", "DegRapXY",  # noqa
                    "RGP", "DynG", "ImPrio", "SzV", "CorThr", "NbPerlm", "L1F", "SatThresh", "Out"]  # noqa

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
        shell_string = f'Tawny {self.mm_args["DataDirectory"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        return shell_string

    def extract_stats(self, raw_output):
        pass

    def validate_mm_parameters(self):
        pass

    def validate_required_files(self):
        pass
