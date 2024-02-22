import _base_command

class AperiCloud(_base_command):

    required_args = ["ImagePattern", "Orientation"]
    allowed_args = ["ImagePattern", "Orientation", "ExpTxt", "Out", "Bin",
                    "RGB", "SeuilEc", "LimBsH", "WithPoints", "CalPerIm",
                    "Focs", "WithCam", "ColCadre", "ColRay", "SH"]

    def __init__(self, args):

        # Initialize the base class
        super().__init__()

        # the input arguments
        self.args = args

        # validate the input arguments
        self.validate_args()