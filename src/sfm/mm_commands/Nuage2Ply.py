# Custom imports
from src.sfm.mm_commands._base_command import BaseCommand

class Nuage2Ply(BaseCommand):
    """
    Convert a point cloud to a ply file
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