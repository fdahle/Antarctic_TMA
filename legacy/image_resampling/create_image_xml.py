from lxml import etree

import base.print_v as p


def create_img_xml(image_id, points, output_folder,
                   catch=True, verbose=False, pbar=None):

    p.print_v(f"Start: create_image_xml ({image_id})", verbose=verbose, pbar=pbar)

    # basic structure of the xml file
    root = etree.Element('SetOfMesureAppuisFlottants')
    maf = etree.SubElement(root, 'MesureAppuiFlottant1Im')

    img_name = etree.SubElement(maf, 'NameIm')
    img_name.text = image_id + ".tif"

    for i in range(8):
        omaf = etree.SubElement(maf, 'OneMesureAF1I')
        pt_name = etree.SubElement(omaf, 'NamePt')
        pt_name.text = "P" + str(i + 1)

        x = int(row["fid_mark_" + str(i + 1) + "_x"])
        y = int(row["fid_mark_" + str(i + 1) + "_y"])

        pt_coords = etree.SubElement(omaf, 'PtIm')
        pt_coords.text = str(x) + " " + str(y)

        tree = etree.ElementTree(root)
        tree.write(output_folder + "/MeasuresIm-" + image_id + '.tif.xml', xml_declaration=True,
                   encoding='UTF-8', pretty_print=True)

    p.print_v(f"Finished: create_image_xml ({image_id})", verbose=verbose, pbar=pbar)
