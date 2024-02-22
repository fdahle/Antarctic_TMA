import _base_command

class ReSampFid(_base_command):

    required_args = ["ImagePattern", "ScanResolution"]
    allowed_args = []

    def __init__(self, args):

        # Initialize the base class
        super().__init__()

        # the input arguments
        self.args = args

        # validate the input arguments
        self.validate_args()