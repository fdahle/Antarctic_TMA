"""Python module for ChgSysCo in Micmac."""

# Library imports
from typing import Any

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class ChgSysCo(BaseCommand):
    """
    Change coordinate system of orientation
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the mm_args
        self.validate_mm_args()

        # validate the input arguments
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

        # TODO
        pass

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
