"""Python module for GrShade in Micmac."""

# Library imports
from typing import Any

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class GrShade(BaseCommand):
    """
    GrShade is used to compute a shading from depth values (or DEM when applicable).
    """

    required_args = ["DEM"]
    allowed_args = ["DEM", "Out", "FileCol", "Visu", "P0", "Sz", "FZ", "DynMed",
                    "Anisotropie", "NbDir", "Brd", "TypeMnt", "TypeShade", "Dequant",
                    "HypsoDyn", "HypsoSat", "SzMaxDalles", "SzRecDalles", "ModeOmbre",
                    "Mask", "DericheFact", "NbIterF", "FactExp", "Dyn", "PdsF",
                    "ModeColor", "NbMed", "NbIterMed", "TetaH", "Azimut"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Initialize the base class
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
        shell_string = f'GrShade {self.mm_args["DEM"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["GrShade"] = shell_string

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

        # TODO

        pass

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

        # TODO
        pass
