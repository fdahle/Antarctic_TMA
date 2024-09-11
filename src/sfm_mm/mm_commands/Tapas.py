"""Python module for Tapas in Micmac."""

# Library imports
import json
import re

# Local imports
from src.sfm_mm.mm_commands._base_command import BaseCommand


class Tapas(BaseCommand):
    """
    Tapas is used for computing purely relative orientations
    """
    required_args = ["DistortionModel", "ImagePattern"]
    allowed_args = ["DistortionModel", "ImagePattern", "ExpTxt", "Out", "InCal", "InOri", "DoC",  # noqa
                    "ForCalib", "Focs", "VitesseInit", "PPRel", "Decentre", "PropDiag", "SauvAutom",  # noqa
                    "ImInit", "MOI"]

    # Tapas has additional arguments for some distortion models
    additional_args = ["DBF", "Debug", "DegRadMax", "DegGen", "LibAff", "LibDec", "LibPP", "LibCP",
                       "LibFoc", "RapTxt", "LinkPPaPPs", "FrozenPoses", "SH", "RefineAll"]
    additional_args_fraser = ["ImMinMax", "EcMax"]
    additional_args_fraser_basic = ["ImMinMax", "EcMax"]
    # noinspection SpellCheckingInspection
    additional_args_fish_eye_equi = ["ImMinMax", "EcMax"]
    # noinspection SpellCheckingInspection
    additional_args_hemi_equi = ["ImMinMax"]

    # noinspection SpellCheckingInspection
    lst_of_distortion_models = ["RadialBasic", "RadialStd", "RadialExtended", "FraserBasic",
                                "Fraser", "FishEyeEqui", "FE_EquiSolBasic", "FishEyeBasic",
                                "FishEyeStereo", "Four", "AddFour", "AddPolyDeg", "Ebner",  # noqa
                                "Brown", "AutoCal", "Figee", "HemiEqui"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        # Initialize the base class with all arguments passed to ReSampFid
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the mm_args
        self.validate_mm_args()

        # validate the input parameters
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
        """

        shell_dict = {}

        # build the basic shell command
        shell_string = f'Tapas {self.mm_args["DistortionModel"]} "{self.mm_args["ImagePattern"]}"'

        # add the optional arguments to the shell string
        for key, val in self.mm_args.items():

            # skip required arguments
            if key in self.required_args:
                continue

            shell_string = shell_string + " " + str(key) + "=" + str(val)

        shell_dict["Tapas"] = shell_string

        return shell_dict

    def extend_additional_args(self) -> None:
        """
        This function builds the shell command.
        """

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

    def extract_stats(self, name: str, raw_output: list[str]) -> None:
        """
        Extract statistics from the raw output of the command and save them to a JSON file.
        Args:
            name (str): Name of the command.
            raw_output (list): Raw output of the command as a list of strings (one per line).
        Returns:
            None
        """

        # initialize the stats dictionary
        stats = {
            "general_info": {
                "total_images": 0,
                "matches": 0,
            },
            "calibration": {},
            "image_stats": {},
            "statistical_summary": [],
            "warnings": {
                "total_warnings": 0,
                "types": []
            }
        }

        # init also stats_dict for later
        stat_dict = {}

        # check if raw_output is a string
        if isinstance(raw_output, str):
            lines = raw_output.split("\n")
        else:
            lines = raw_output

        def _save_float_convert(val):
            try:
                return float(val)
            except ValueError:
                return val

        # iterate all lines
        for line in lines:
            # get number of images
            if '"OIS.*tif":' in line:
                stats["general_info"]["total_images"] = int(re.search(r'(\d+) matches', line).group(1))

            # get number of matches

            if "MdPppppF=" in line:
                stats["calibration"] = {
                    "FocMm": float(re.search(r'FocMm(\d+\.\d+)', line).group(1)),
                    "XSZ": [int(x) for x in re.findall(r'XSZ=\[(\d+),(\d+)]', line)[0]],
                }

            # get values per image
            if "RES:" in line:

                # get the image name
                image_name = re.search(r'RES:\[(.+?)]\[C]', line).group(1)

                # create a new entry for the image if it does not exist
                if image_name not in stats["image_stats"]:
                    stats["image_stats"][image_name] = {}
                    stats["image_stats"][image_name]["error"] = []  # noqa
                    stats["image_stats"][image_name]["percentage_residuals"] = []  # noqa
                    stats["image_stats"][image_name]["nr_tps"] = []  # noqa
                    stats["image_stats"][image_name]["multiple_points"] = []  # noqa
                    stats["image_stats"][image_name]["multiple_points_res"] = []  # noqa
                    stats["image_stats"][image_name]["time"] = []  # noqa

                # extract values
                error = _save_float_convert(re.search(r'ER2 ([\d.-]+)', line).group(1))
                perc_residuals = _save_float_convert(re.search(r'Nn ([\d.]+)', line).group(1))
                nr_tps = int(re.search(r'Of ([\d.]+)', line).group(1))
                multiple_points = int(re.search(r'Mul ([\d.]+)', line).group(1))
                multiple_points_res = int(re.search(r'Mul-NN ([\d.]+)', line).group(1))
                time = _save_float_convert(re.search(r'Time ([\d.]+)', line).group(1))

                # special case for residu
                if error == "-":
                    error = "-nan"

                # save values to the dictionary
                stats["image_stats"][image_name]["error"].append(error)  # noqa
                stats["image_stats"][image_name]["percentage_residuals"].append(perc_residuals)  # noqa
                stats["image_stats"][image_name]["nr_tps"].append(nr_tps)  # noqa
                stats["image_stats"][image_name]["multiple_points"].append(multiple_points)  # noqa
                stats["image_stats"][image_name]["multiple_points_res"].append(multiple_points_res)  # noqa
                stats["image_stats"][image_name]["time"].append(time)  # noqa

            # new stat dict required
            if "Stat on type of point" in line:
                stat_dict = {
                    "Step": 0,
                    "Iteration": 0,
                    "PdResNull": 0,
                    "PdResNullPerc": 0,
                    "Behind": 0,
                    "BehindPerc": 0,
                    "VisibIm": 0,
                    "VisibImPerc": 0,
                }

            # save the values to stat dict
            if "*   Perc=" in line:

                match = re.search(r'Perc=(\d+\.\d+)% ; {2}Nb=(\d+) for (\w+)', line)
                stat_dict[match.group(3)] = int(match.group(2))
                stat_dict[f"{match.group(3)}Perc"] = _save_float_convert(match.group(1))

            # save the stat dict
            if line.startswith("--- End Iter"):

                # get step and iter
                pattern = r"Iter (\d+) STEP (\d+)"
                match = re.search(pattern, line)
                iter_value = int(match.group(1))
                step_value = int(match.group(2))

                # save both values
                stat_dict["Iteration"] = iter_value
                stat_dict["Step"] = step_value

                # append dict to the list
                stats["statistical_summary"].append(stat_dict)

            if "warnings of" in line:
                stats["warnings"]["total_warnings"] = int(re.search(r'There were (\d+) warnings', line).group(1))

        # Serialize and save the extracted stats
        json_output = json.dumps(stats, indent=4)
        with open(f"{self.project_folder}/stats/{name}_stats.json", "w") as file:
            file.write(json_output)

    def validate_mm_parameters(self) -> None:
        """
        Validate the input parameters of the command.
        """

        if self.mm_args["DistortionModel"] not in self.lst_of_distortion_models:
            raise ValueError(f"DistortionModel {self.mm_args['DistortionModel']} is not a valid model.")

    def validate_required_files(self) -> None:
        """
        Validate the required files of the command.
        """

        # TODO

        pass
