"""
This function returns the subset of single image
"""
import sys #in order to import other python files
import cv2 #in order to load the image
import matplotlib.pyplot as plt # in order to display stuff
import numpy as np #for random shuffeling

#add for import
sys.path.append('../../misc')

from database_connections import *
from image_loader import *

verbose = True
selection_mode = "random" #possible values are 'all', 'random', 'id' (then
                              #you must provide an id)
imageId = "CA135832V0065" #for selection mode 'id'
only_complete=False #for selection mode 'all' and 'random'

#extract the subsets from a image for the specified directions
def get_subset(imgId, display=False, fid_directions=["N", "S", "E", "W"]):

    #get the required data from the table
    def get_table_data(imgId):

        if verbose:
            print("get data from database..", end='\r')

        conn = Connector()

        #specify params for table select
        tables = ["image_properties", "images"]
        fields = [
                  ["photo_id",
                   "subset_width", "subset_height",
                   "subset_N_x", "subset_N_y",
                   "subset_E_x", "subset_E_y",
                   "subset_S_x", "subset_S_y",
                   "subset_W_x", "subset_W_y"],
                   ["file_path"]
                 ]
        join = ["photo_id"]
        filters = [
                   {"image_properties.photo_id": imgId}
                  ]

        #get data from table
        tableData = conn.get_data(tables, fields, filters, join)

        if verbose:
            print("get data from database.. - finished")

        return tableData

    def extract_subset(img, x, y, width, height):
        if y < 0:
            y = 0
        if x < 0:
            x = 0
        subset = img[y:y+height, x:x+width]
        return subset

    #get table information and load image
    tableData = get_table_data(imgId)
    img = load_image(tableData["file_path"][0])

    #create subsetdict that will save the subsets
    subsets = {}

    #extract subset for each direction
    for fid_direction in fid_directions:

        #get params
        x = tableData["subset_"+fid_direction+"_x"][0]
        y = tableData["subset_"+fid_direction+"_y"][0]
        width = tableData["subset_width"][0]
        height = tableData["subset_height"][0]


        # the actual substractions
        if x == None or y == None:
            subsets[fid_direction] = None
        else:
            subsets[fid_direction] = extract_subset(img, x,y,width,height)


    if display:
        f, axes = plt.subplots(2,2)
        if subsets["N"] is not None:
            axes[0,0].imshow(subsets["N"], cmap='gray')
            axes[0,0].set_title("N")

        if subsets["E"] is not None:
            axes[0,1].imshow(subsets["E"], cmap='gray')
            axes[0,1].set_title("E")

        if subsets["S"] is not None:
            axes[1,0].imshow(subsets["S"], cmap='gray')
            axes[1,0].set_title("S")

        if subsets["W"] is not None:
            axes[1,1].imshow(subsets["W"], cmap='gray')
            axes[1,1].set_title("W")

        f.suptitle(imgId)

        #that plots are not overlapping
        f.tight_layout()

        plt.show()

    return subsets

#get ids for selection type all and random
def get_ids_from_table(bool_complete):

    if verbose:
        print("reading ids from table..", end='\r')

    conn = Connector()

    #specify params for table select
    tables = ["image_properties"]
    fields = [["photo_id"]]

    if bool_complete:
        filters = [
            {"image_properties.subset_n_x":"NOT NULL",
             "image_properties.subset_n_y":"NOT NULL",
             "image_properties.subset_e_x":"NOT NULL",
             "image_properties.subset_e_y":"NOT NULL",
             "image_properties.subset_s_x":"NOT NULL",
             "image_properties.subset_s_y":"NOT NULL",
             "image_properties.subset_w_x":"NOT NULL",
             "image_properties.subset_w_y":"NOT NULL"}
        ]
    else:
        filters = [
            {"image_properties.subset_n_x":"NOT NULL",
             "image_properties.subset_n_y":"NOT NULL"},
            {"image_properties.subset_e_x":"NOT NULL",
             "image_properties.subset_e_y":"NOT NULL"},
            {"image_properties.subset_s_x":"NOT NULL",
             "image_properties.subset_s_y":"NOT NULL"},
            {"image_properties.subset_w_x":"NOT NULL",
             "image_properties.subset_w_y":"NOT NULL"}
        ]

    #get data from table
    tableData = conn.get_data(tables, fields, filters)

    if verbose:
        print("reading ids from table.. - finished")

    return tableData.values

if __name__ == "__main__":

    if selection_mode == "id":
        if imageId is None:
            raise ValueError("Please provide an imageId with this mode")
        get_subset(imageId, display=True)

    else:
        ids = get_ids_from_table(only_complete)

        if selection_mode == "random":
            np.random.shuffle(ids)

        for elem in ids:
            get_subset(elem[0], display=True)
