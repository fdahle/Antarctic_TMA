import json
import re

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

    def extract_stats(self, raw_output):
        stats = {
            "general_info": {
                "total_images": 0,
                "matches": 0,
            },
            "calibration": {},
            "image_processing": [],
            "statistical_summary": [],
            "warnings": {
                "total_warnings": 0,
                "types": []
            }
        }

        lines = raw_output.split("\n")
        for line in lines:
            if '"OIS.*tif":' in line:
                stats["general_info"]["total_images"] = int(re.search(r'(\d+) matches', line).group(1))
            if "MdPppppF=" in line:
                stats["calibration"] = {
                    "FocMm": float(re.search(r'FocMm(\d+\.\d+)', line).group(1)),
                    "XSZ": [int(x) for x in re.findall(r'XSZ=\[(\d+),(\d+)\]', line)[0]],
                }
            if "RES:" in line:
                image_name = re.search(r'RES:\[(.+?)\]\[C\]', line).group(1)
                residu = float(re.search(r'ER2 ([\d\.-]+)', line).group(1))
                time = float(re.search(r'Time ([\d\.]+)', line).group(1))
                stats["image_processing"].append({"name": image_name, "residu": residu, "time": time})
            if "Stat on type of point" in line or "Perc=" in line:
                match = re.search(r'Perc=(\d+\.\d+)% ;  Nb=(\d+) for (\w+)', line)
                if match:
                    stats["statistical_summary"].append({
                        "type": match.group(3),
                        "percentage": float(match.group(1)),
                        "number": int(match.group(2)),
                    })
            if "warnings of" in line:
                stats["warnings"]["total_warnings"] = int(re.search(r'There were (\d+) warnings', line).group(1))

        # Serialize and save the extracted stats
        json_output = json.dumps(stats, indent=4)
        with open("tapas_stats.json", "w") as file:
            file.write(json_output)

    def validate_mm_parameters(self):

        if self.mm_args["DistortionModel"] not in self.lst_of_distortion_models:
            raise ValueError(f"DistortionModel {self.mm_args['DistortionModel']} is not a valid model.")

    def validate_required_files(self):
        pass


