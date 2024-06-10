"""Python module for ReduceCustom (custom function) in Micmac."""

# Package imports
import glob
import json

# Custom imports
import src.load.load_image as li
import src.export.export_tiff as et
from src.sfm_mm.mm_commands._base_command import BaseCommand
from src.sfm_mm.mm_commands._context_manager import log_and_print


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
        """
        This function is called before the execution of the command.
        """
        # nothing needs to be done before the execution
        pass

    def after_execution(self):
        """
        This function is called after the execution of the command.
        """
        # nothing needs to be done after the execution
        pass

    def build_shell_dict(self):
        """
        This function builds the shell command.
        """
        raise AssertionError("This custom class does not have a shell command.")

    def execute_custom_cmd(self):

        # Redirect stdout to capture printed output
        with log_and_print() as log_stream:

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

                # Save the original shape
                old_shape = img.shape

                # Remove the border from the image
                img = img[border:height - border, border:width - border]

                # Save the new shape
                new_shape = img.shape

                print(f"Shape reduced from {old_shape} to {new_shape} for image {image_path}")

                # Save the image
                et.export_tiff(img, image_path, overwrite=True)

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

    def extract_stats(self, name, raw_output):
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

        stats={}
        print("TODO")

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # save json_output to a file
        with open(f"{self.project_folder}/stats/{name}_stats.json", "w") as file:
            file.write(json_output)

    def validate_mm_parameters(self):
        """
        Validate the input parameters of the command.
        """

        # TODO
        pass

    def validate_required_files(self):
        """
        Validate the required files of the command.
        """

        # TODO

        pass
