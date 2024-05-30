# Package imports
import glob


# Custom imports
import src.load.load_image as li
import src.export.export_tiff as et
from src.sfm_mm.mm_commands._base_command import BaseCommand


class ReduceCustom(BaseCommand):
    required_args = ["Pxl"]
    allowed_args = ["Pxl"]

    def __init__(self, *args, **kwargs):

        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # validate the mm_args
        self.validate_mm_args()

        # validate the input parameters
        self.validate_mm_parameters()

    def before_execution(self):
        # nothing needs to be done before the execution
        pass

    def after_execution(self):
        # nothing needs to be done after the execution
        pass

    def build_shell_dict(self):
        raise AssertionError("This custom class does not have a shell command.")

    def execute_custom_cmd(self):
        # get all tif files in project folder that start with "OIS-"
        images = glob.glob(f"{self.project_folder}/OIS-*.tif")

        # add all images that are in the folder mask
        images += glob.glob(f"{self.project_folder}/masks/*.tif")

        for image_path in images:
            # Load the image
            img = li.load_image(image_path)

            # Get the size of the image
            height, width = img.shape

            # Define the size of the border to remove
            border = int(self.mm_args["Pxl"])

            # Remove the border from the image
            img = img[border:height - border, border:width - border]

            # Save the image
            et.export_tiff(img, image_path, overwrite=True)

    def extract_stats(self, name, raw_output):
        pass

    def validate_mm_parameters(self):
        pass

    def validate_required_files(self):
        pass
