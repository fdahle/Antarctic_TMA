# Package imports
import pandas as pd
from lxml import etree


def create_image_xml(image_id: str, fid_marks: pd.DataFrame, output_path: str) -> None:
    """
    Create an XML file with fiducial mark coordinates for resampling images with MicMac.
    Args:
        image_id (str): The ID of the image.
        fid_marks (pd.DataFrame): DataFrame containing fiducial mark coordinates.
        output_path (str): The file path where the XML will be saved.
    Returns:
        None
    """

    # basic structure of the xml file
    root = etree.Element('SetOfMesureAppuisFlottants')
    maf = etree.SubElement(root, 'MesureAppuiFlottant1Im')

    img_name = etree.SubElement(maf, 'NameIm')
    img_name.text = image_id + ".tif"

    for i in range(8):
        omaf = etree.SubElement(maf, 'OneMesureAF1I')
        pt_name = etree.SubElement(omaf, 'NamePt')
        pt_name.text = "P" + str(i + 1)

        x = int(fid_marks["fid_mark_" + str(i + 1) + "_x"].iloc[0])
        y = int(fid_marks["fid_mark_" + str(i + 1) + "_y"].iloc[0])

        pt_coords = etree.SubElement(omaf, 'PtIm')
        pt_coords.text = str(x) + " " + str(y)

    tree = etree.ElementTree(root)
    tree.write(output_path, xml_declaration=True,
               encoding='UTF-8', pretty_print=True)
