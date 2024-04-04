import os
import glob
import xml.etree.ElementTree as ET

from lxml import etree

import src.load.load_image as li
import src.load.load_transform as lt

import src.sfm.snippets.identify_gpcs as ig

from src.sfm.mm_commands._base_command import BaseCommand


class GCPCustom(BaseCommand):

    required_args = []
    allowed_args = ["ALLTransformsReq"]

    def __init__(self, *args, **kwargs):

        # Initialize the base class
        super().__init__(*args, **kwargs)

        # save the input arguments
        self.args = args
        self.kwargs = kwargs

        # set default values for the mm_args
        if "ALLTransformsReq" not in self.mm_args:
            self.mm_args["ALLTransformsReq"] = True

    def build_shell_string(self):
        raise AssertionError("This custom class does not have a shell command.")

    def execute_custom_cmd(self):

        # validate the required files
        self.validate_required_files(self.mm_args["ALLTransformsReq"])

        # get the gcps
        gcp_dict = self._get_gcps()

        # resample the gcps
        gcp_dict = self._resample_gcps(gcp_dict)

        # create the xml files
        self._create_measures_xml(gcp_dict, self.project_folder)
        self._create_measures_2d_xml(gcp_dict, self.project_folder)

    def extract_stats(self, raw_output):
        pass

    def validate_required_files(self, all_transforms_req):
        # TODO
        pass


    def _get_gcps(self):

        # get all image_ids from the images folder
        file_paths = glob.glob(self.project_folder + "/images_orig/*.tif")
        image_ids = [os.path.basename(file)[:-4] for file in file_paths]

        # load the images
        images = []
        for image_id in image_ids:
            img = li.load_image(image_id, self.project_folder + "/images_orig")
            images.append(img)

        # load the transforms from the images
        transforms = []
        for image_id in image_ids:
            transform = lt.load_transform(image_id, self.project_folder + "/transforms")
            transforms.append(transform)

        # get identical gcps for the images
        gcp_dict = ig.identify_gcps(image_ids, images, transforms)

        return gcp_dict

    def _resample_gcps(self, gcp_dict):

        import src.sfm.snippets.resample_tie_points as rtp
        import src.sfm.snippets.calc_resample_matrix as crm

        image_transforms = {}

        for point, data in gcp_dict.items():
            new_rel_coords = []
            for image_id, rel_coords in zip(data['image_ids'], data['rel_coords']):

                # load image transform if not existing yet
                if image_id not in image_transforms.keys():
                    image_xml = crm.calc_resample_matrix(self.project_folder, image_id)
                    image_transforms[image_id] = image_xml
                else:
                    image_xml = image_transforms[image_id]

                # get the transformed coordinates
                transformed_coords = rtp.resample_tie_points(rel_coords, image_xml)

                # convert back to tuple
                transformed_coords = tuple(transformed_coords[0])

                new_rel_coords.append(transformed_coords)

            # Update the dictionary with the new relative coordinates
            data['rel_coords'] = new_rel_coords

        return gcp_dict

    @staticmethod
    def _create_measures_xml(gcp_dict, save_fld):

        root = etree.Element("DicoAppuisFlottant")

        for i, g_dict in enumerate(gcp_dict.values()):
            gcp_element = etree.SubElement(root, "OneAppuisDAF")

            abs_coords = etree.SubElement(gcp_element, "Pt")
            abs_coords.text = str(g_dict["avg_abs_coord"][0]) + " " + \
                              str(g_dict["avg_abs_coord"][1])

            gcp_name = etree.SubElement(gcp_element, "NamePt")
            gcp_name.text = "GCP" + str(i + 1)

            reliability = etree.SubElement(gcp_element, "Incertitude")
            reliability.text = "1 1"

        tree = etree.ElementTree(root)
        tree.write(save_fld + "/Measures.xml",
                   xml_declaration=True, encoding='utf-8', pretty_print=True)

    @staticmethod
    def _create_measures_2d_xml(gcp_dict, save_fld):

        root = etree.Element("SetOfMesureAppuisFlottants")

        # this dict contains the xml elements for each image
        image_xml_dict = {}

        for i, gcp_dict in enumerate(gcp_dict.values()):

            for j in range(len(gcp_dict["image_ids"])):

                # get the image id and rel coords
                image_id = gcp_dict["image_ids"][j]
                coords = gcp_dict["rel_coords"][j]

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
                rel_coords.text = str(coords[0]) + " " + str(coords[1])

        tree = etree.ElementTree(root)
        tree.write(save_fld + "/Measures-S2D.xml",
                   xml_declaration=True, encoding='utf-8', pretty_print=True)

