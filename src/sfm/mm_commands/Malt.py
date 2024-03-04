from src.sfm.mm_commands._base_command import BaseCommand

class Malt(BaseCommand):
    """
    Compute the DEM
    """

    required_args = ["Mode", "ImagePattern", "Orientation"]
    allowed_args = ["Mode", "ImagePattern", "Orientation", "Master", "SzW", "CorMS", "UseGpu", "Regul",
                    "DirMEC", "DirOF", "UseTA", "ZoomF", "ZoomI", "ZPas", "Exe", "Repere", "NbVI", "HrOr",
                    "LrOr", "DirTA", "Purge", "DoMEC", "DoOrtho", "UnAnam", "2Ortho", "ZInc", "DefCor",
                    "CostTrans", "Etape0", "AffineLast", "ResolOrtho", "ImMNT", "ImOrtho", "ZMoy",
                    "Spherik", "WMI", "vMasqIm", "MasqImGlob", "IncMax", "BoxClip", "BoxTerrain", "ResolTerrain",
                    "RoundResol", "GCC", "EZA", "Equiv", "MOri", "MaxFlow", "SzRec", "Masq3D",
                    "NbProc", "PSIBN", "InternalNoIncid", "PtDebug"]

    lst_of_modes = ["Ortho", "UrbanMNE", "GeomImage"]


    def __init__(self, *args, **kwargs):
        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

    def build_shell_string(self):

        # build the basic shell command
        shell_string = f'Malt {self.mm_args["Mode"]} {self.mm_args["ImagePattern"]} ' \
                       f'{self.mm_args["Orientation"]}'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        return shell_string

    def validate_mm_parameters(self):

        if "/" in self.mm_args["ImagePattern"]:
            raise ValueError("ImagePattern cannot contain '/'. Use a pattern like '*.tif' instead.")