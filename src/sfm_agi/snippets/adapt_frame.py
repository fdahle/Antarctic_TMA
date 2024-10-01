import os
import xml.etree.ElementTree as ET
import zipfile

import src.sfm_agi.snippets.zip_folder as zp


def adapt_frame(project_files, element_tag, attribute_name, attribute_value):
    """

    Args:
        project_files:
        element_tag:
        attribute_name:
        attribute_value:

    Returns:

    """
    # adapt path to the frame folder
    path_frame_fld = os.path.join(project_files, "0", "0")
    path_frame_file = os.path.join(path_frame_fld, "frame.zip")
    path_frame_xml = os.path.join(path_frame_fld, "frame", "doc.xml")

    # unzip the frame
    with zipfile.ZipFile(path_frame_file, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(path_frame_fld, "frame"))

    # Load the XML file
    tree = ET.parse(path_frame_xml)
    root = tree.getroot()

    # Find the element with the specific tag
    element_found = False
    for elem in root.iter(element_tag):
        # Add or update the attribute to the element
        elem.set(attribute_name, attribute_value)
        element_found = True

    # If the element doesn't exist, you can create it and insert it
    if not element_found:
        new_element = ET.Element(element_tag)
        new_element.set(attribute_name, attribute_value)
        root.append(new_element)

    ET.dump(root)  # This will print the entire XML structure to the console

    # Save the changes back to the XML file
    tree.write(path_frame_xml)

    print(f"Write frame to {path_frame_xml}")

    zp.zip_folder(os.path.join(path_frame_fld, "frame"),
                  path_frame_file, delete_files=True)

    if os.path.exists(os.path.join(path_frame_fld, "frame")):
        os.rmdir(os.path.join(path_frame_fld, "frame"))
