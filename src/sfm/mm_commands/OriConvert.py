from src.sfm.mm_commands._base_command import BaseCommand


class OriConvert(BaseCommand):
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
        pass

    def after_execution(self):
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
        pass

    def validate_mm_parameters(self):
        pass

    def validate_required_files(self):
        pass