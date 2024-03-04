import os
import glob
import numpy as np

#
import src.base.find_overlapping_images as foi
import src.base.find_tie_points as ftp

#
import src.load.load_image as li

#
from src.sfm.mm_commands._base_command import BaseCommand

debug = True

class TapiocaCustom(BaseCommand):

    required_args = []
    allowed_args = ["use_footprints", "use_masks"]

    def __init__(self, *args, **kwargs):

        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the input arguments
        self.validate_mm_parameters()

    def build_shell_string(self):
        raise AssertionError("This custom class does not have a shell command.")

    def execute_custom_cmd(self):

        # validate the required files
        self.validate_required_files()

        # create the tie point structure
        self._create_tie_point_structure()

    def validate_mm_parameters(self):
        print("Validate mm parameters")

        pass

    def validate_required_files(self):
        print("Validate required files")

        if 'use_masks' in self.mm_args.keys() and self.mm_args['use_masks'] is True:

            # store all missing mask ids here
            missing_masks = []

            # check for each file if a mask exists
            tif_files = glob.glob(self.project_folder + "/*.tif")
            for file in tif_files:
                base_name = os.path.basename(file)
                base_name = base_name[:-4]
                mask_path = self.project_folder + "/masks/" + base_name + ".tif"
                if os.path.isfile(mask_path) is False:
                    missing_masks.append(base_name)

            if len(missing_masks) > 0:
                raise FileNotFoundError(f"{len(missing_masks)} masks are missing: {missing_masks}")

    def _create_tie_point_structure(self):

        # create homol if not existing
        if os.path.isdir(self.project_folder + "/Homol") is False:
            os.mkdir(self.project_folder + "/Homol")

        # get all image_ids from the images folder
        file_paths = glob.glob(self.project_folder + "/images/*.tif")
        file_names = [os.path.basename(file)[:-4] for file in file_paths]

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

        if len(short_file_names) < 2:
            raise ValueError(f"{len(short_file_names)} images are available in the images folder. "
                             f"At least 2 images are required for tie-point matching.")

        if debug:
            print(f"{len(short_file_names)} images are tested for overlap")

        # find overlapping images
        # todo: add support for footprints
        short_overlap_dict = foi.find_overlapping_images(short_file_names, working_modes=["ids"])

        # convert the short names to the full names again
        overlap_dict = {prefix + key: [prefix + value for value in values] for key, values in short_overlap_dict.items()}

        # save the loaded images here so that we don't need to load always again
        loaded_images = {}
        loaded_masks = {}

        # init the tie point detector
        tpd = ftp.TiePointDetector("lightglue", catch=False)

        # store checked combinations of images
        combinations = {}

        # find tie points between overlapping images
        for key_id, other_ids in overlap_dict.items():

            print(f"{key_id} is overlapping with {len(other_ids)} other images")

            # check if the key image is already loaded
            if key_id in loaded_images:
                key_image = loaded_images[key_id]
            else:
                key_image = li.load_image(key_id, image_path=self.project_folder + "/images")
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
                    other_image = li.load_image(other_id, self.project_folder + "/images")
                    loaded_images[other_id] = other_image

                if 'use_masks' in self.mm_args.keys() and self.mm_args['use_masks'] is True:
                    if other_id in loaded_masks:
                        other_mask = loaded_masks[other_id]
                    else:
                        other_mask = li.load_image(other_id, self.project_folder + "/masks")
                        loaded_masks[other_id] = other_mask
                else:
                    other_mask = None

                # find tie points between the key and the other image
                tps, conf = tpd.find_tie_points(key_image, other_image,
                                                key_mask, other_mask)

                conf = conf.reshape(-1, 1)

                # merge tps and quality
                tps_c = np.concatenate((tps, conf), axis=1)

                print(f"{tps_c.shape[0]} tie points found between {key_id} and {other_id}")

                # continue if no tie points are found
                if tps.shape[0] == 0:
                    continue

                # create the folder
                path_key_folder = self.project_folder + "/Homol/Pastis" + key_id + ".tif"
                if os.path.isdir(path_key_folder) is False:
                    os.mkdir(path_key_folder)

                print(path_key_folder + "/" + key_id + "tif.txt")

                np.savetxt(path_key_folder + "/" + other_id + ".tif.txt", tps_c,
                           fmt=['%i', '%i', '%.i', '%.i', '%.3f'], delimiter=" ")
