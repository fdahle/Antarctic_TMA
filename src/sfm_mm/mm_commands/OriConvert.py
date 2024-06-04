# Package imports
import json
import os

# Custom imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class OriConvert(BaseCommand):
    """
    OriConvert is a versatile command used to:
    - Transform embedded GPS data from text format to MicMac's Xml orientation format.
    - Transform the GPS coordinate system, potentially into a euclidean coordinate system.
    - Generate image pattern for selecting a sample of the image block.
    - Compute relative speed of each camera in order to determine and correct GPS systematic error (delay).
    - Importing external orientation from others software: to come.
    """

    required_args = ["FormatSpecification", "OrientationFile", "TargetedOrientation"]
    allowed_args = ["FormatSpecification", "OrientationFile", "TargetedOrientation",
                    "ChSys", "Calib", "AddCalib", "ConvOri", "PrePost", "KN2I", "DN", "ImC",
                    "NbImC", "RedSizeSC", "Reexp", "Regul", "RegNewBr", "Reliab", "CalcV",
                    "Delay", "TFC", "RefOri", "SiftR", "SiftLR", "NameCple", "Delaunay",
                    "DelaunayCross", "Cpt", "UOC", "MTD1", "Line", "CBF", "AltiSol", "Prof",
                    "OffsetXY", "CalOFC", "OkNoIm", "SzW"]

    def __init__(self, *args, **kwargs):

        # Initialize the base class with all arguments passed to OriConvert
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
        shell_string = f'OriConvert {self.mm_args["FormatSpecification"]} ' \
                       f'{self.mm_args["OrientationFile"]} ' \
                       f'{self.mm_args["TargetedOrientation"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["OriConvert"] = shell_string

        return shell_dict

    def extract_stats(self, name, raw_output):

        # Initialize statistics dictionary
        stats = {}

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # Define path to save the json file
        json_path = f"{self.project_folder}/stats/{name}_stats.json"

        # Save json_output to a file
        with open(json_path, "w") as file:
            file.write(json_output)

        if self.debug:
            print(f"OriConvert: Stats saved to {json_path}")

    def validate_mm_parameters(self):
        pass

    def validate_required_files(self):

        if self.mm_args["FormatSpecification"] == "OriTxtInFile" and \
                os.path.isfile(self.project_folder + "/" + self.mm_args["OrientationFile"]) is False:
            raise FileNotFoundError(f"OrientationFile '{self.mm_args['OrientationFile']}' is missing.")
