"""
This algorithm creates the xml files for where the fid points of the camera are
"""

#algorithm params
xmlFolder = "../../data/xml/cameras_xml"

verbose = True

import sys #in order to import other python files
from lxml import etree as ET #to create xml files
import numpy as np

#add for import
sys.path.append('../misc')

from database_connections import *


def create_cam_xml():

    conn = Connector()

    def get_table_data():

        if verbose:
            print("get data from database..", end='\r')

        tables = ["cameras"]
        fields = "*"

        #get data from table
        tableData = conn.get_data(tables, fields)

        if verbose:
            print("get data from database.. - finished")

        return tableData

    def convert_coordinates(row):

        #micmac counts from top left, wheras the coordinates in the sql table
        #are based from the ppc
        #3  7  2
        #
        #5 PPA 6
        #
        #1  8  4

        #get min x and y value
        x_min = np.min([row["fid_1_x"],row["fid_2_x"],row["fid_3_x"],row["fid_4_x"],
                  row["fid_5_x"],row["fid_6_x"],row["fid_7_x"],row["fid_8_x"]])
        y_min = np.min(-1 * np.array([row["fid_1_y"],row["fid_2_y"],row["fid_3_y"],row["fid_4_y"],
                  row["fid_5_y"],row["fid_6_y"],row["fid_7_y"],row["fid_8_y"]]))

        row["fid_1_x"] = row["fid_1_x"] - x_min
        row["fid_1_y"] = - row["fid_1_y"] - y_min
        row["fid_2_x"] = row["fid_2_x"] - x_min
        row["fid_2_y"] = - row["fid_2_y"] - y_min
        row["fid_3_x"] = row["fid_3_x"] - x_min
        row["fid_3_y"] = - row["fid_3_y"] - y_min
        row["fid_4_x"] = row["fid_4_x"] - x_min
        row["fid_4_y"] = - row["fid_4_y"] - y_min
        row["fid_5_x"] = row["fid_5_x"] - x_min
        row["fid_5_y"] = - row["fid_5_y"] - y_min
        row["fid_6_x"] = row["fid_6_x"] - x_min
        row["fid_6_y"] = - row["fid_6_y"] - y_min
        row["fid_7_x"] = row["fid_7_x"] - x_min
        row["fid_7_y"] = - row["fid_7_y"] - y_min
        row["fid_8_x"] = row["fid_8_x"] - x_min
        row["fid_8_y"] = - row["fid_8_y"] - y_min

        return row

    def create_xml(row):

        root = ET.Element('MesureAppuiFlottant1Im')

        camName = ET.SubElement(root, 'NameIm')
        camName.text = row["description"]
        coordinates = [1,2,3,4,5,6,7,8]

        for elem in coordinates:
            oneMes = ET.SubElement(root, 'OneMesureAF1I')
            ptName = ET.SubElement(oneMes, 'NamePt')
            ptName.text = "P" + str(elem)

            x = row["fid_" + str(elem) + "_x"]
            y = row["fid_" + str(elem) + "_y"]

            x = np.round(x, 4)
            y = np.round(y, 4)

            ptCoordinates = ET.SubElement(oneMes, 'PtIm')
            ptCoordinates.text = str(x) + "\t " + str(y)

        tree = ET.ElementTree(root)
        tree.write(xmlFolder + "/" + row["description"] + "-" + 'MeasuresCamera.xml', pretty_print=True)

    camData = get_table_data()

    for idx, row in camData.iterrows():

        if verbose:
            print("create xml for " + row["description"] + "..", end='\r')

        row = convert_coordinates(row)

        create_xml(row)

        if verbose:
            print("create xml for " + row["description"] + ".. - finished", end='\r')

if __name__ == "__main__":

    create_cam_xml()
