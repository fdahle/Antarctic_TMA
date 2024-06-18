"""Python module for TapiocaCustom (custom function) in Micmac."""

# Library imports
import json
import os
import glob
import numpy as np
import re
from typing import Any

# Local imports
import src.base.find_overlapping_images as foi
import src.base.find_tie_points as ftp
import src.load.load_image as li
from src.sfm_mm.mm_commands._base_command import BaseCommand
from src.sfm_mm.mm_commands._context_manager import log_and_print


class TapiocaCustom(BaseCommand):
    """
    TapiocaCustom is a custom command used for computing tie points
    between overlapping images using SuperGlue/LightGlue.
    """

    required_args = []
    allowed_args = ["use_footprints", "use_masks", "max_id_range", "save_tps_images"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:

        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input arguments
        self.validate_mm_parameters()

    def before_execution(self) -> None:
        """
        This function is called before the execution of the command.
        """

        # nothing required before execution
        pass

    def after_execution(self) -> None:
        """
        This function is called after the execution of the command.
        """

        # nothing required after execution
        pass

    def build_shell_dict(self) -> None:
        """
        This function would normally build the shell command, but is not needed for this custom class.
        Raises:
            AssertionError: Because this custom class does not have a shell command.
        """
        raise AssertionError("This custom class does not have a shell command.")

    def execute_custom_cmd(self) -> None:
        """
        This function executes the custom functions of the command.
        """

        # validate the required files
        self.validate_required_files()

        # Redirect stdout to capture printed output
        with log_and_print() as log_stream:

            # create the tie point structure
            self._create_tie_point_structure()

        # extract the log output
        raw_output = log_stream.getvalue()

        # save the raw output to a file
        if self.save_raw:
            filename = f"{self.project_folder}/stats/" \
                       f"{self.command_name}_raw.txt"
            with open(filename, "w") as file:
                file.write(raw_output)

        if self.save_stats:
            self.extract_stats(self.command_name, raw_output)

    def extract_stats(self, name: str, raw_output: list[str]) -> None:
        """
        Extract statistics from the raw output of the command and save them to a JSON file.
        Args:
            name (str): Name of the command.
            raw_output (list): Raw output of the command as a list of strings (one per line).
        Returns:
            None
        """

        # Split the raw_output into lines if it's a single string
        if isinstance(raw_output, str):
            raw_output = raw_output.splitlines()

        # Initialize statistics dictionary
        stats = {
            "general": {
                "total_images": 0,
                "total_tie_points": 0,
            },
            "images": []
        }

        total_tie_points = 0
        current_image = None

        # Iterate over each line to extract and organize information
        for line in raw_output:
            if re.match(r'^\d+ images are tested for overlap:', line):
                total_images = int(re.findall(r'\d+', line)[0])
                stats["general"]["total_images"] = total_images
            elif re.match(r'^\s*OIS-Reech_', line):
                parts = line.split()
                current_image = {
                    "name": parts[0],
                    "overlapping_images": []
                }
                stats["images"].append(current_image)
            elif re.match(r'^\s+\d+ tie points found between', line):
                parts = re.findall(r'\d+|\bOIS-Reech_\w+\b', line)
                tie_points = int(parts[0])
                other_image = parts[2]

                total_tie_points += tie_points

                current_image["overlapping_images"].append({
                    "other_image": other_image,
                    "tie_points": tie_points
                })

        stats["general"]["total_tie_points"] = total_tie_points

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # save json_output to a file
        with open(f"{self.project_folder}/stats/{name}_stats.json", "w") as file:
            file.write(json_output)

    def validate_mm_parameters(self) -> None:
        """
        Validate the input parameters of the command.
        """

        print("Validate mm parameters", end='')

        if "save_tps_images" not in self.mm_args.keys():
            self.mm_args["save_tps_images"] = False

        if "max_id_range" not in self.mm_args.keys():
            self.mm_args["max_id_range"] = 1

        print("\rValidate mm parameters - finished")

    def validate_required_files(self) -> None:
        """
        Validate the required files of the command.
        """

        if self.debug:
            print("Validate required files", end='')

        if 'use_masks' in self.mm_args.keys() and self.mm_args['use_masks'] is True:

            # store all missing mask ids here
            missing_masks = []

            # get all image files
            tif_files = glob.glob(self.project_folder + "/*.tif")

            # check if at least 2 images are available
            if len(tif_files) < 2:
                raise FileNotFoundError(f"{len(tif_files)} images are available. "
                                        f"At least 2 images are required for tie-point matching.")

            # check for each file if a mask exists
            for file in tif_files:
                base_name = os.path.basename(file)
                base_name = base_name[:-4]
                mask_path = self.project_folder + f"/masks/{base_name}.tif"
                if os.path.isfile(mask_path) is False:
                    missing_masks.append(base_name)

            if len(missing_masks) > 0:
                raise FileNotFoundError(f"{len(missing_masks)} masks are missing: {missing_masks}")

        if self.debug:
            print("\rValidate required files - finished")

    def _create_tie_point_structure(self):

        # create homol if not existing
        if os.path.isdir(self.project_folder + "/Homol") is False:
            os.mkdir(self.project_folder + "/Homol")

        # get all image_ids from the project folder
        file_paths = glob.glob(self.project_folder + "/*.tif")
        file_names = [os.path.basename(file)[:-4] for file in file_paths]

        # prefix for the images
        prefix = "OIS-Reech_"

        # also get the short file names for finding overlapping images
        short_file_names = []
        for file_name in file_names:
            if file_name.startswith(prefix):
                # Remove the prefix
                short_file_name = file_name[len(prefix):]
            else:
                # If there's no prefix, use the original file name
                short_file_name = file_name
            short_file_names.append(short_file_name)

        if self.debug:
            print(f"{len(short_file_names)} images are tested for overlap:")

        # find overlapping images
        # todo: add support for footprints
        short_overlap_dict = foi.find_overlapping_images(short_file_names, working_modes=["ids"],
                                                         max_id_range=self.mm_args["max_id_range"])

        # convert the short names to the full names again
        overlap_dict = {prefix + key: [prefix + value for value in values] for
                        key, values in short_overlap_dict.items()}

        # save the loaded images here so that we don't need to load always again
        loaded_images = {}
        loaded_masks = {}

        # init the tie point detector
        tpd = ftp.TiePointDetector("lightglue", catch=False)

        # find tie points between overlapping images
        for key_id, other_ids in overlap_dict.items():

            if self.debug:
                print(f" {key_id} is overlapping with {len(other_ids)} other images")

            # check if the key image is already loaded
            if key_id in loaded_images:
                key_image = loaded_images[key_id]
            else:
                key_image = li.load_image(key_id, image_path=self.project_folder)
                loaded_images[key_id] = key_image

            if 'use_masks' in self.mm_args.keys() and self.mm_args['use_masks'] is True:
                if key_id in loaded_masks:
                    key_mask = loaded_masks[key_id]
                else:
                    key_mask = li.load_image(key_id, self.project_folder + "/masks")
                    loaded_masks[key_id] = key_mask
            else:
                key_mask = None

            # iterate all overlapping images
            for other_id in other_ids:

                # check if the other image is already loaded
                if other_id in loaded_images:
                    other_image = loaded_images[other_id]
                else:
                    other_image = li.load_image(other_id, self.project_folder)
                    loaded_images[other_id] = other_image

                if 'use_masks' in self.mm_args.keys() and self.mm_args['use_masks'] is True:
                    if other_id in loaded_masks:
                        other_mask = loaded_masks[other_id]
                    else:
                        other_mask = li.load_image(other_id, self.project_folder + "/masks")
                        loaded_masks[other_id] = other_mask
                else:
                    other_mask = None

                if self.mm_args["save_tps_images"]:

                    tp_img_folder = self.project_folder + "/visuals/tps"

                    if os.path.isdir(tp_img_folder) is False:
                        os.mkdir(tp_img_folder)

                    save_path = tp_img_folder + "/" + key_id + "_" + other_id + ".png"
                else:
                    save_path = None

                # find tie points between the key and the other image
                tps, conf = tpd.find_tie_points(key_image, other_image,
                                                key_mask, other_mask,
                                                save_path=save_path)

                conf = conf.reshape(-1, 1)

                # merge tps and quality
                tps_c = np.concatenate((tps, conf), axis=1)

                print(f"  {tps_c.shape[0]} tie points found between {key_id} and {other_id}")

                # continue if no tie points are found
                if tps.shape[0] == 0:
                    continue

                # create the folder
                path_key_folder = self.project_folder + "/Homol/Pastis" + key_id + ".tif"
                if os.path.isdir(path_key_folder) is False:
                    os.mkdir(path_key_folder)

                # save the tie points as txt file
                np.savetxt(path_key_folder + "/" + other_id + ".tif.txt", tps_c,
                           fmt=['%i', '%i', '%.i', '%.i', '%.3f'], delimiter=" ")

        print("Tie point structure created successfully.")
