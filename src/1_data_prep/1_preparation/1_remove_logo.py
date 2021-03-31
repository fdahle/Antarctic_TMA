"""
This algorithm checks if the image is oriented upside down or not
"""

import sys #in order to import other python files
import cv2 #in order for some cv methods
import mahotas as mht
import numpy as np

import matplotlib.pyplot as plt


#add for import
sys.path.append('../../misc')

from database_connections import *
from image_loader import *

#debug params
verbose=True


def remove_logo():

    conn = Connector()

    def get_table_data():

        if verbose:
            print("get data from database..", end='\r')

        tables = ["image_properties", "images"]
        fields = [["photo_id","logo_removed"],
                  ["file_path"]
                  ]
        join = ["photo_id"]
        filters = [
            {"logo_removed":"FALSE()"}
        ]

        #get data from table
        tableData = conn.get_data(tables, fields, filters, join)

        if verbose:
            print("get data from database.. - finished")

        return tableData

    def substract_logo(img):
        img = img[0:img.shape[0] - 350,:]
        return img

    def save_image(img, imgPath):
        try:
            cv2.imwrite(imgPath, img)
            return True
        except:
            return False

    def update_table(conn, photo_id):

        table="image_properties"
        id={"photo_id":photo_id}
        data={"logo_removed":"TRUE()"}

        conn.edit_data(table, id, data)

    #get the data from the database
    tableData = get_table_data()

    for idx, row in tableData.iterrows():

        if verbose:
            print("remove logo for " + row["photo_id"] + "..", end='\r')

        #load the images
        img, imgPath = load_image(row['file_path'], get_path = True)

        if verbose and img is None:
            print("remove logo for " + row["photo_id"] + ".. - Image not found")
            continue

        img = substract_logo(img)

        saved = save_image(img, imgPath)

        if saved:
            update_table(conn, row["photo_id"])
        else:
            print("remove logo for " + row["photo_id"] + ".. - could not save Image")
            continue

        if verbose:
            print("remove logo for " + row["photo_id"] + ".. - finished sucessfully")


if __name__ == "__main__":
    remove_logo()
