# Package imports
import os
import glob
import json
import numpy as np
from lxml import etree

# Custom imports
import src.load.load_image as li
import src.load.load_transform as lt
import src.sfm_mm.snippets.calc_resample_matrix as crm
import src.sfm_mm.snippets.identify_gpcs as ig
import src.sfm_mm.snippets.resample_tie_points as rtp
from src.sfm_mm.mm_commands._base_command import BaseCommand
from src.sfm_mm.mm_commands._context_manager import log_and_print


class GCPCustom(BaseCommand):
    required_args = []
    allowed_args = ["ALLTransformsReq", "UseMasks"]

    def __init__(self, *args, **kwargs):

        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # set default values for the mm_args
        if "ALLTransformsReq" not in self.mm_args:
            self.mm_args["ALLTransformsReq"] = True

        if "UseMasks" not in self.mm_args:
            self.mm_args["UseMasks"] = False

        # validate the mm_args
        self.validate_mm_args()

        # validate the input parameters
        # self.validate_mm_parameters()

    def before_execution(self):
        # nothing needs to be done before the execution
        pass

    def after_execution(self):
        # nothing needs to be done after the execution
        pass

    def build_shell_dict(self):
        raise AssertionError("This custom class does not have a shell command.")

    def execute_custom_cmd(self):

        # validate the required files
        self.validate_required_files()

        # Redirect stdout to capture printed output
        with log_and_print() as log_stream:

            # get the gcps
            gcp_dict = self._get_gcps()

            # resample the gcps
            gcp_dict = self._resample_gcps(gcp_dict)

            # create the xml files
            self._create_measures_xml(gcp_dict, self.project_folder)
            self._create_measures_2d_xml(gcp_dict, self.project_folder)

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

        # Split the raw_output into lines if it's a single string
        if isinstance(raw_output, str):
            raw_output = raw_output.splitlines()

        stats = {}
        print("TODO")

        # Serialize the dictionary to a JSON string
        json_output = json.dumps(stats, indent=4)

        # save json_output to a file
        with open(f"{self.project_folder}/stats/{name}_stats.json", "w") as file:
            file.write(json_output)

    def validate_required_files(self):

        # for each image we need a transform file
        images = glob.glob(self.project_folder + "/images_orig/*.tif")

        missing_transforms = []

        for image in images:

            # get image id from the image path
            image_id = os.path.basename(image)[:-4]

            # get path to transform file
            path_transform_file = f"{self.project_folder}/transforms/{image_id}.txt"

            # check if the transform file exists
            if not os.path.isfile(path_transform_file):
                missing_transforms.append(image_id)

        if len(missing_transforms) == len(images):
            raise FileNotFoundError(f"No transform files found in '{self.project_folder}/transforms'.")

        # in this case we need all transforms
        if self.mm_args["ALLTransformsReq"] and len(missing_transforms) > 0:
            if len(missing_transforms) == 1:
                error_str = f"{len(missing_transforms)} transform file is missing."
            else:
                error_str = f"{len(missing_transforms)} transform files are missing."
            raise FileNotFoundError(error_str)

        # check if we should use masks
        if self.mm_args["UseMasks"]:

            # store all missing masks in this list
            missing_masks = []

            # iterate all images
            for image in images:

                # get image id from the image path
                image_id = os.path.basename(image)[:-4]

                # get path to mask file
                path_mask_file = f"{self.project_folder}/masks_orig/{image_id}.tif"

                # check if the mask file exists
                if not os.path.isfile(path_mask_file):
                    missing_masks.append(image_id)

            if len(missing_masks) > 0:
                if len(missing_masks) == len(images):
                    error_str = f"No mask files found in '{self.project_folder}/masks'."
                elif len(missing_masks) == 1:
                    error_str = f"{len(missing_masks)} mask file is missing."
                else:
                    error_str = f"{len(missing_masks)} mask files are missing."
                raise FileNotFoundError(error_str)

    def _get_gcps(self):

        # get all image_ids from the images folder
        file_paths = glob.glob(self.project_folder + "/images_orig/*.tif")
        image_ids = [os.path.basename(file)[:-4] for file in file_paths]

        # load the images
        images = []
        for image_id in image_ids:
            if self.debug:
                print(f"GCPCustom: Load image '{image_id}'")

            img = li.load_image(image_id, self.project_folder + "/images_orig")
            images.append(img)

        # load the transforms from the images
        transforms = []
        for image_id in image_ids:
            if self.debug:
                print(f"GCPCustom: Load transform '{image_id}'")
            try:
                transform = lt.load_transform(image_id, self.project_folder + "/transforms")
                transforms.append(transform)
            # ignore if the transform file does not exist
            except (Exception,):
                transforms.append(np.zeros((3, 3)))

        # load the masks if needed
        if self.mm_args["UseMasks"]:
            masks = []
            for image_id in image_ids:
                if self.debug:
                    print(f"GCPCustom: Load mask '{image_id}'")

                mask = li.load_image(image_id, self.project_folder + "/masks_orig")
                masks.append(mask)
        else:
            masks = None

        # get identical gcps for the images
        gcp_dict = ig.identify_gcps(image_ids, images, transforms, masks)

        return gcp_dict

    def _resample_gcps(self, gcp_dict):
        """
         convert the gcps to the absolute coords
        """

        image_transforms = {}

        for point, data in gcp_dict.items():

            for entry in data:

                image_id = entry["image_id"]
                rel_coords = (entry["x"], entry["y"])

                # load image transform if not existing yet
                if image_id not in image_transforms.keys():
                    image_trans_mat = crm.calc_resample_matrix(self.project_folder, image_id)
                    image_transforms[image_id] = image_trans_mat
                else:
                    image_trans_mat = image_transforms[image_id]

                # get the transformed coordinates
                transformed_coords = rtp.resample_tie_points(rel_coords, image_trans_mat)

                if np.any(transformed_coords < 0):
                    raise ValueError(f"Negative x coordinate found in transformed points for {image_id}.")

                # convert back to tuple
                transformed_coords = tuple(transformed_coords[0])

                entry["x"] = transformed_coords[0]
                entry["y"] = transformed_coords[1]

        return gcp_dict

    @staticmethod
    def _create_measures_xml(gcp_dict, save_fld):

        root = etree.Element("DicoAppuisFlottant")

        # iterate through all gcps
        for i, key in enumerate(gcp_dict.keys()):

            # get the gcp entry
            gcp_entry = gcp_dict[key]

            # create the gcp element in xml
            gcp_element = etree.SubElement(root, "OneAppuisDAF")

            # get the absolute coords
            abs_coords = etree.SubElement(gcp_element, "Pt")
            abs_coords.text = str(key[0]) + " " + str(key[1]) + " " + str(gcp_entry[0]["z_abs"])

            # create the name of the gcp
            gcp_name = etree.SubElement(gcp_element, "NamePt")
            gcp_name.text = "GCP" + str(i + 1)

            # set reliability of the gcp
            reliability = etree.SubElement(gcp_element, "Incertitude")
            reliability.text = "100 100 100"

        tree = etree.ElementTree(root)
        tree.write(save_fld + "/Measures.xml",
                   xml_declaration=True, encoding='utf-8', pretty_print=True)

    @staticmethod
    def _create_measures_2d_xml(gcp_dict, save_fld):

        root = etree.Element("SetOfMesureAppuisFlottants")

        # this dict contains the xml elements for each image
        image_xml_dict = {}

        for i, gcp_entry in enumerate(gcp_dict.values()):

            for j in range(len(gcp_entry)):

                # get the image id and rel coords
                image_id = gcp_entry[j]["image_id"]
                x = gcp_entry[j]["x"]
                y = gcp_entry[j]["y"]

                # check if image already has a xml element
                if image_id not in image_xml_dict.keys():
                    image_xml_element = etree.SubElement(root, "MesureAppuiFlottant1Im")
                    image_name = etree.SubElement(image_xml_element, "NameIm")
                    image_name.text = "OIS-Reech_" + image_id + ".tif"
                    image_xml_dict[image_id] = image_xml_element  # Save the reference
                else:
                    image_xml_element = image_xml_dict[image_id]

                # create the gcp element
                gcp_element = etree.SubElement(image_xml_element, "OneMesureAF1I")

                gcp_name = etree.SubElement(gcp_element, "NamePt")
                gcp_name.text = "GCP" + str(i + 1)

                rel_coords = etree.SubElement(gcp_element, "PtIm")
                rel_coords.text = str(x) + " " + str(y)

        tree = etree.ElementTree(root)
        tree.write(save_fld + "/Measures-S2D.xml",
                   xml_declaration=True, encoding='utf-8', pretty_print=True)
