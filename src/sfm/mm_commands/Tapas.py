import _base_command

class Tapas(_base_command):

    required_args = ["DistortionModel", "ImagePattern"]
    allowed_args = ["DistortionModel", "ImagePattern", "ExpTxt", "Out", "InCal", "InOri", "DoC",  # noqa
                    "ForCalib", "Focs", "VitesseInit", "PPRel", "Decentre", "PropDiag", "SauvAutom",  # noqa
                    "ImInit", "MOI"]

    def __init__(self, args):

        # Initialize the base class
        super().__init__()

        # the input arguments
        self.args = args

        # validate the input arguments
        self.validate_args()