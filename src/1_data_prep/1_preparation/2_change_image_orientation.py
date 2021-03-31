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


def change_image_orientation():

    conn = Connector()

    def get_table_data():

        if verbose:
            print("get data from database..", end='\r')

        tables = ["image_properties", "images"]
        fields = [["photo_id",
                   "line_e_x1", "line_e_x2",
                   "line_w_x1", "line_w_x2"
                  ],
                  ["file_path"]
                  ]
        join = ["photo_id"]
        filters = [
            {"image_properties.right_orientation": "FALSE()"}
        ]

        #get data from table
        tableData = conn.get_data(tables, fields, filters, join)

        if verbose:
            print("get data from database.. - finished")

        return tableData

    def extract_image_parts(img, data):
        image_parts = {"E": None, "W": None}

        width = 250

        for key in image_parts:

            if key == "E":
                image_parts[key] = img[:, img.shape[1] - width:]
            elif key == "W":
                image_parts[key] = img[:,0:width]

        return image_parts

    def check_orientation(image_parts):

        contrast={"E":None, "W":None}

        for key in contrast:
            blurred = cv2.GaussianBlur(image_parts[key],(9,9),0)

            ret,fid_th = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            #calculate haralick features
            hara = mht.features.haralick(fid_th).mean(0)

            #contrast; higher contrast = lower homogeneity
            contrast[key] = hara[1]

        orientation = None

        if contrast["E"] < contrast["W"]:
            orientation = "correct"
        elif contrast["W"] < contrast["E"]:
            orientation = "upside down"

        return orientation

    def change_orientation(photo_id, img):

        img = np.rot90(img, 2)

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
        data={"right_orientation":"TRUE()"}

        conn.edit_data(table, id, data)


    #get the data from the database
    tableData = get_table_data()

    for idx, row in tableData.iterrows():

        if verbose:
            print("change orientation for " + row["photo_id"] + "..", end='\r')

        #load the images
        img, imgPath = load_image(row['file_path'], get_path = True)

        if verbose and img is None:
            print("check orientation for " + row["photo_id"] + ".. - Image not found")
            continue

        image_parts = extract_image_parts(img, row)

        orientation = check_orientation(image_parts)

        if verbose and orientation is None:
            print("check orientation for " + row["photo_id"] + ".. - no orientation found")
            continue

        if orientation == "upside down":
            img = change_orientation(row["photo_id"], img)
            saved = save_image(img, imgPath)
        else:
            saved = True

        if saved:
            update_table(conn, row["photo_id"])
        else:
            print("change orientation for " + row["photo_id"] + ".. - could not save Image")
            continue

        if verbose:
            print("change orientation for " + row["photo_id"] + ".. - finished sucessfully")


if __name__ == "__main__":
    change_image_orientation()
