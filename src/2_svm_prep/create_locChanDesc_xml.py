"""
This algorithm creates the xml files with the parameters of the camera
"""

#algorithm params
xmlFolder = "../../data/xml/cameras_xml"

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

        tables = ["cameras"]
        fields = "*"

        #get data from table
        tableData = conn.get_data(tables, fields)

        if verbose:
            print("get data from database.. - finished")

        return tableData


    def create_xml(data):
        root = ET.Element('Global')
        chDesc = ET.SubElement(root, 'ChantierDescripteur')

        #define camera model
        locCamDataBase = ET.SubElement(chDesc, 'LocCamDataBase')
        camEntry = ET.SubElement(locCamDataBase, 'CameraEntry')

        camName = ET.SubElement(camEntry, 'Name')
        camName.text = data["description"]

        #Size of sensore (digital) or for analog,
        #scanned film : MidSideFiducials or "MaxFidX-MinFidX MaxFidY-MinFidY
        szCaptMm = ET.SubElement(camEntry, 'SzCaptMm')
        szCaptMm.text = str(data["midside_fid_x"]) + " " + str(data["midside_fid_y"])

        shortName = ET.SubElement(camEntry, 'ShortName')
        shortName.text = data["description"]

        ###
        #for the camera model
        ###
        keyNameAsso = ET.SubElement(chDesc, 'KeyedNamesAssociations')
        calcs = ET.SubElement(keyNameAsso, 'Calcs')

        arrite = ET.SubElement(calcs, 'Arrite')
        arrite.text = "1 1"

        direct = ET.SubElement(calcs, 'Direct')

        patTrans = ET.SubElement(direct, 'PatternTransform')
        patTrans.text = ".*"

        CalcName = ET.SubElement(direct, 'CalcName')
        CalcName.text = data["description"]

        key = ET.SubElement(keyNameAsso, "Key")
        key.text = "NKS-Assoc-STD-CAM"

        ###
        #for the focal length
        ###
        keyNameAsso = ET.SubElement(chDesc, 'KeyedNamesAssociations')
        calcs = ET.SubElement(keyNameAsso, 'Calcs')

        arrite = ET.SubElement(calcs, 'Arrite')
        arrite.text = "1 1"

        direct = ET.SubElement(calcs, 'Direct')

        patTrans = ET.SubElement(direct, 'PatternTransform')
        patTrans.text = ".*"

        CalcName = ET.SubElement(direct, 'CalcName')
        CalcName.text = str(data["calibrated_focal_length"])

        key = ET.SubElement(keyNameAsso, "Key")
        key.text = "NKS-Assoc-STD-FOC"

        tree = ET.ElementTree(root)
        tree.write(xmlFolder + "/" + row["description"] + "-" + 'LocalChantierDescripteur.xml', pretty_print=True)


    camData = get_table_data()

    for idx, row in camData.iterrows():

        if verbose:
            print("create xml for " + row["description"] + "..", end='\r')

        create_xml(row)

        if verbose:
            print("create xml for " + row["description"] + ".. - finished")


if __name__ == "__main__":

    create_loc_chan_desc()
