"""Python module for SaisieMasq in Micmac."""


# Custom imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class SaisieMasq(BaseCommand):
    """
    SaisieMasq is a very simple tool to edit mask images.
    It creates a binary mask image from a polygonal selection in the displayed image.
    """
    required_args = []
    allowed_args = []

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

    def before_execution(self):
        # nothing needs to be done before the execution
        pass

    def after_execution(self):
        # nothing needs to be done after the execution
        pass

    def build_shell_dict(self):

        shell_dict = {}

        # build the basic shell command
        shell_string = f'SaisieMasq '

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["SaisieMasq"] = shell_string

        return shell_dict

    def extract_stats(self, name, raw_output):
        pass

    def validate_mm_parameters(self):
        pass

    def validate_required_files(self):
        pass
