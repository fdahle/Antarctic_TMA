from src.sfm.mm_commands._base_command import BaseCommand

class Tapas(BaseCommand):

    required_args = ["DistortionModel", "ImagePattern"]
    allowed_args = ["DistortionModel", "ImagePattern", "ExpTxt", "Out", "InCal", "InOri", "DoC",  # noqa
                    "ForCalib", "Focs", "VitesseInit", "PPRel", "Decentre", "PropDiag", "SauvAutom",  # noqa
                    "ImInit", "MOI"]

    # Tapas has additional arguments for some distortion models
    additional_args = ["DBF", "Debug", "DegRadMax", "DegGen", "LibAff", "LibDec", "LibPP", "LibCP",
                       "LibFoc", "RapTxt", "LinkPPaPPs", "FrozenPoses", "SH", "RefineAll"]
    additional_args_fraser = ["ImMinMax", "EcMax"]
    additional_args_fraser_basic = ["ImMinMax", "EcMax"]
    additional_args_fish_eye_equi = ["ImMinMax", "EcMax"]
    additional_args_hemi_equi = ["ImMinMax"]

    lst_of_distortion_models = ["RadialBasic", "RadialStd", "RadialExtended", "FraserBasic",
                                "Fraser", "FishEyeEqui", "FE_EquiSolBasic", "FishEyeBasic",
                                "FishEyeStereo", "Four", "AddFour", "AddPolyDeg", "Ebner",  # noqa
                                "Brown", "AutoCal", "Figee", "HemiEqui"]

    def __init__(self, *args, **kwargs):

        # Initialize the base class with all arguments passed to ReSampFid
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input parameters
        self.validate_mm_parameters()

    def build_shell_string(self):

        # build the basic shell command
        shell_string = f'Tapas {self.mm_args["DistortionModel"]} "{self.mm_args["ImagePattern"]}"'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        return shell_string

    def extend_additional_args(self):

        # extend allowed arguments based on distortion model
        if self.mm_args["DistortionModel"] == "RadialBasic":
            self.allowed_args = self.allowed_args + self.additional_args
        elif self.mm_args["DistortionModel"] == "RadialExtended":
            self.allowed_args = self.allowed_args + self.additional_args
        elif self.mm_args["DistortionModel"] == "Fraser":
            self.allowed_args = self.allowed_args + self.additional_args + self.additional_args_fraser
        elif self.mm_args["DistortionModel"] == "FraserBasic":
            self.allowed_args = self.allowed_args + self.additional_args + self.additional_args_fraser_basic
        elif self.mm_args["DistortionModel"] == "FishEyeEqui":
            self.allowed_args = self.allowed_args + self.additional_args + self.additional_args_fish_eye_equi
        elif self.mm_args["DistortionModel"] == "HemiEqui":
            self.allowed_args = self.allowed_args + self.additional_args + self.additional_args_hemi_equi


    def validate_mm_parameters(self):

        if self.mm_args["DistortionModel"] not in self.lst_of_distortion_models:
            raise ValueError(f"DistortionModel {self.mm_args['DistortionModel']} is not a valid model.")

    def validate_required_files(self):
        pass


