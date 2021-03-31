"""
This algorithm creates the xml files for each images which tells the coordinates
of the fid points
"""

#algorithm params
xmlFolder = "../../data/xml/images_xml"

#debug params
verbose = True

import sys #in order to import other python files

#add for import
sys.path.append('../misc')

from database_connections import *




from lxml import etree as ET


def create_loc_chan_desc():

    conn = Connector()

    def get_table_data():

        if verbose:
            print("get data from database..", end='\r')

        tables = ["image_properties"]

        fields = [["photo_id",
                  "fid_mark_1_x", "fid_mark_1_y", "fid_mark_1_estimated",
                  "fid_mark_2_x", "fid_mark_2_y", "fid_mark_2_estimated",
                  "fid_mark_3_x", "fid_mark_3_y", "fid_mark_3_estimated",
                  "fid_mark_4_x", "fid_mark_4_y", "fid_mark_4_estimated",
                  "fid_mark_5_x", "fid_mark_5_y", "fid_mark_5_estimated",
                  "fid_mark_6_x", "fid_mark_6_y", "fid_mark_6_estimated",
                  "fid_mark_7_x", "fid_mark_7_y", "fid_mark_7_estimated",
                  "fid_mark_8_x", "fid_mark_8_y", "fid_mark_8_estimated"]]


        filters = [
            {"image_properties.fid_mark_1_x":"NOT NULL",
             "image_properties.fid_mark_1_y":"NOT NULL",
             "image_properties.fid_mark_2_x":"NOT NULL",
             "image_properties.fid_mark_2_y":"NOT NULL",
             "image_properties.fid_mark_3_x":"NOT NULL",
             "image_properties.fid_mark_3_y":"NOT NULL",
             "image_properties.fid_mark_4_x":"NOT NULL",
             "image_properties.fid_mark_4_y":"NOT NULL",
             "image_properties.fid_mark_5_x":"NOT NULL",
             "image_properties.fid_mark_5_y":"NOT NULL",
             "image_properties.fid_mark_6_x":"NOT NULL",
             "image_properties.fid_mark_6_y":"NOT NULL",
             "image_properties.fid_mark_7_x":"NOT NULL",
             "image_properties.fid_mark_7_y":"NOT NULL",
             "image_properties.fid_mark_8_x":"NOT NULL",
             "image_properties.fid_mark_8_y":"NOT NULL"
             }
        ]

        #get data from table
        tableData = conn.get_data(tables, fields, filters)

        if verbose:
            print("get data from database.. - finished")

        return tableData

    def create_xml(imageData):

        #extract data from the row
        imgId = imageData["photo_id"]

        #basic structure of the xml file
        root = ET.Element('SetOfMesureAppuisFlottants')
        maf = ET.SubElement(root, 'MesureAppuiFlottant1Im')

        imgName = ET.SubElement(maf, 'NameIm')
        imgName.text = imgId + ".tif"

        for i in range(8):
            omaf = ET.SubElement(maf, 'OneMesureAF1I')
            ptName = ET.SubElement(omaf, 'NamePt')
            ptName.text = "P" + str(i+1)

            x = imageData["fid_mark_" + str(i+1) +"_x"]
            y = imageData["fid_mark_" + str(i+1) +"_y"]

            ptCoords = ET.SubElement(omaf, 'PtIm')
            ptCoords.text = str(x) + " " + str(y)

        tree = ET.ElementTree(root)
        tree.write(xmlFolder + "/MeasuresIm-" + imgId + '.tif.xml', xml_declaration=True,
         encoding='UTF-8', pretty_print=True)

    tableData = get_table_data()

    for idx, row in tableData.iterrows():

        if verbose:
            print("create xml for " + row["photo_id"] + "..", end='\r')

        create_xml(row)

        if verbose:
            print("create xml for " + row["photo_id"] + ".. - finished")


if __name__ == "__main__":

    create_loc_chan_desc()
