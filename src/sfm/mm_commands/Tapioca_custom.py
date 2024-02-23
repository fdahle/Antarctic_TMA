import src.base.find_overlapping_images as foi
import src.base.find_tie_points as ftp

import src.load.load_image as li

class Tapioca_custom():

    required_args = ["ImagePattern", "ScanResolution"]
    allowed_args = []

    def __init__(self, args):

        # Initialize the base class
        super().__init__()

        # the input arguments
        self.args = args

        # validate the input arguments
        self.validate_args()


    def create_tie_point_structure(self):

        # create homol if not existing
        if os.path.isdir(self.project_folder + "/Homol") is False:
            os.mkdir(self.project_folder + "/Homol")

        # get all image_ids from the images folder
        tif_files = glob.glob(self.project_folder + "/*.tif")

        # find overlapping images
        overlap_dict = foi.find_overlapping_images(tif_files)

        loaded_images = {}

        # find tie points between overlapping images
        for key_id, other_ids in overlap_dict.items():

            if key_id in loaded_images:
                key_image = loaded_images[key_id]
            else:
                key_image = li.load_image(key_id)
                loaded_images[key_id] = key_image

            for other_id in other_ids:

                if other_id in loaded_images:
                    other_image = loaded_images[other_id]
                else:
                    other_image = li.load_image(other_id)
                    loaded_images[other_id] = other_image

                tps = ftp.find_tie_points(key_image, other_image)