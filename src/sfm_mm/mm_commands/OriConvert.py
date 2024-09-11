"""Python module for OriConvert in Micmac."""

# Library imports
import json
import os
from typing import Any

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class OriConvert(BaseCommand):
    """
    OriConvert is a versatile command used to:
    - Transform embedded GPS data from text format to MicMac's Xml orientation format.
    - Transform the GPS coordinate system, potentially into a Euclidean coordinate system.
    - Generate image pattern for selecting a sample of the image block.
    - Compute relative speed of each camera in order to determine and correct GPS systematic error (delay).
    - Importing external orientation from others software: to come.
    """

    required_args = ["FormatSpecification", "OrientationFile", "TargetedOrientation"]
    # noinspection SpellCheckingInspection
    allowed_args = ["FormatSpecification", "OrientationFile", "TargetedOrientation",
                    "ChSys", "Calib", "AddCalib", "ConvOri", "PrePost", "KN2I", "DN", "ImC",
                    "NbImC", "RedSizeSC", "Reexp", "Regul", "RegNewBr", "Reliab", "CalcV",
                    "Delay", "TFC", "RefOri", "SiftR", "SiftLR", "NameCple", "Delaunay",
                    "DelaunayCross", "Cpt", "UOC", "MTD1", "Line", "CBF", "AltiSol", "Prof",
                    "OffsetXY", "CalOFC", "OkNoIm", "SzW"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        # Initialize the base class with all arguments passed to OriConvert
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the mm_args
        self.validate_mm_args()

        # validate the input parameters
        self.validate_mm_parameters()

    def before_execution(self) -> None:
        """
        This function is called before the execution of the command.
        """

        # nothing needs to be done before the execution
        pass

    def after_execution(self) -> None:
        """
        This function is called after the execution of the command.
        """

        # nothing needs to be done after the execution
        pass

    def build_shell_dict(self) -> dict[str, str]:
        """
        This function builds the shell command.
        Returns:
            dict[str, str]: Dictionary containing the command name and the command string.
        """

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

    def extract_stats(self, name: str, raw_output: list[str]) -> None:
        """
        Extract statistics from the raw output of the command and save them to a JSON file.
        Args:
            name (str): Name of the command.
            raw_output (list): Raw output of the command as a list of strings (one per line).
        Returns:
            None
        """

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

    def validate_mm_parameters(self) -> None:
        """
        Validate the input parameters of the command.
        """

        # TODO

        pass

    def validate_required_files(self) -> None:
        """
        Validate the required files of the command.
        """

        if self.mm_args["FormatSpecification"] == "OriTxtInFile" and \
                os.path.isfile(self.project_folder + "/" + self.mm_args["OrientationFile"]) is False:
            raise FileNotFoundError(f"OrientationFile '{self.mm_args['OrientationFile']}' is missing.")
