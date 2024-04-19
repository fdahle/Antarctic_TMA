import xml.etree.ElementTree as ET

path = "/data_1/ATM/data_1/sfm/projects/EGU2/Measures.xml"

PT_VAL = 0
INCERTITUDE_VAL = 0.05

def three_dfy_xml(xml_path, pt_val, inc_val):
    # Load the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterate over each OneAppuisDAF element
    for one_appui in root.findall('OneAppuisDAF'):
        # Update Pt by appending a third dimension
        pt = one_appui.find('Pt')
        if pt is not None:
            pt.text += f' {pt_val}'  # Append the third dimension value as a string

        # Update Incertitude by appending a third dimension
        incertitude = one_appui.find('Incertitude')
        if incertitude is not None:
            incertitude.text += f' {inc_val}'  # Append the third dimension value as a string

    # Save the modified XML to a new file
    new_path = xml_path.replace('.xml', '_3d.xml')
    tree.write(new_path, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":

    three_dfy_xml(path, PT_VAL, INCERTITUDE_VAL)
