"""Python module for Nuage2Ply in Micmac."""

# Package Imports
import glob
import os

# Custom imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class Nuage2Ply(BaseCommand):
    """
    Nuage2Ply is used to convert a depth map (or DEM when applicable) to a point cloud in ply format.
    The color of an image can be projected to the points (option "attr").
    In case of a DEM (classically computed by "Malt Ortho"), the ortho-image mosaic computed by
    Tawny can be used
    """

    required_args = ["XmlFile"]
    allowed_args = ["XmlFile", "Sz", "P0", "Out", "Scale", "Attr", "Comments", "Bin", "Mask",
                    "SeuilMask", "Dyn", "DoPly", "DoXYZ", "Normale", "NormByC", "ExagZ",
                    "RatioAttrCarte", "Mesh", "64B", "Offs", "NeighMask", "ForceRGB"]

    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

    def before_execution(self):
        """
        This function is called before the execution of the command.
        """

        if self.mm_args["XmlFile"] == "<AUTO>":

            input_fld = os.path.join(self.project_folder, "MEC-Malt")

            xml_pattern = os.path.join(input_fld, "NuageImProf_STD*.xml")
            xml_files = glob.glob(xml_pattern)

            if xml_files:
                # Get the most recent file by modification time
                most_recent_file = max(xml_files, key=os.path.getmtime)
                self.mm_args["XmlFile"] = "MEC-Malt/" + os.path.basename(most_recent_file)
            else:
                raise FileNotFoundError("No XML file found")

    def after_execution(self):
        """
        This function is called after the execution of the command.
        """
        # nothing needs to be done after the execution
        pass

    def build_shell_dict(self):
        """
        This function builds the shell command.
        """

        shell_dict = {}

        # build the basic shell command
        shell_string = f'Nuage2Ply {self.mm_args["XmlFile"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["Nuage2Ply"] = shell_string

        return shell_dict

    def extract_stats(self, name, raw_output):
        """
        Extract statistics from the raw output of the command and save them to a JSON file.
        Args:
            name (str): Name of the command.
            raw_output (list): Raw output of the command as a list of strings (one per line).
        Returns:
            None
        """

        pass

    def validate_mm_parameters(self):
        """
        Validate the input parameters of the command.
        """

        pass

    def validate_required_files(self):
        """
        Validate the required files of the command.
        """

        pass
