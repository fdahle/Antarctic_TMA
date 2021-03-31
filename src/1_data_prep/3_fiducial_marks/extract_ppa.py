"""
This algorithm calculate the ppa of an image based either on corner or side
fiducial points
"""

import sys #in order to import other python files
import math #to check for nan
import cv2 #in order for some cv methods

import matplotlib.pyplot as plt

#add for import
sys.path.append('../../misc')

from database_connections import *
from image_loader import *

#debug params
verbose=True

debug_show_ppa = True

def extract_ppa():

    conn = Connector()

    def get_table_data():

        if verbose:
            print("get data from database..", end='\r')

        tables = ["image_properties", "images"]
        fields = [["photo_id",
                   "fid_mark_1_x", "fid_mark_1_y", "fid_mark_1_estimated",
                   "fid_mark_2_x", "fid_mark_2_y", "fid_mark_2_estimated",
                   "fid_mark_3_x", "fid_mark_3_y", "fid_mark_3_estimated",
                   "fid_mark_4_x", "fid_mark_4_y", "fid_mark_4_estimated",
                   "fid_mark_5_x", "fid_mark_5_y", "fid_mark_5_estimated",
                   "fid_mark_6_x", "fid_mark_6_y", "fid_mark_6_estimated",
                   "fid_mark_7_x", "fid_mark_7_y", "fid_mark_7_estimated",
                   "fid_mark_8_x", "fid_mark_8_y", "fid_mark_8_estimated"
                  ],
                  ["file_path"]
                  ]
        join=["photo_id"]
        filters = [
            {"image_properties.fid_mark_1_x":"NOT NULL",
             "image_properties.fid_mark_1_y":"NOT NULL",
             "image_properties.fid_mark_2_x":"NOT NULL",
             "image_properties.fid_mark_2_y":"NOT NULL",
             "image_properties.fid_mark_3_x":"NOT NULL",
             "image_properties.fid_mark_3_y":"NOT NULL",
             "image_properties.fid_mark_4_x":"NOT NULL",
             "image_properties.fid_mark_4_y":"NOT NULL"},
            {"image_properties.fid_mark_5_x":"NOT NULL",
             "image_properties.fid_mark_5_y":"NOT NULL",
             "image_properties.fid_mark_6_x":"NOT NULL",
             "image_properties.fid_mark_6_y":"NOT NULL",
             "image_properties.fid_mark_7_x":"NOT NULL",
             "image_properties.fid_mark_7_y":"NOT NULL",
             "image_properties.fid_mark_8_x":"NOT NULL",
             "image_properties.fid_mark_8_y":"NOT NULL"}
        ]

        #get data from table
        tableData = conn.get_data(tables, fields, filters, join)

        if verbose:
            print("get data from database.. - finished")

        return tableData

    def calculate_ppa(points):

        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]

        x3, y3 = points[2][0], points[2][1]
        x4, y4 = points[3][0], points[3][1]


        D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

        ppa_x = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / D

        ppa_y = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/D

        return [ppa_x, ppa_y]

    def compare_ppas(ppa_list):

        distance = math.sqrt( ((ppa_list[0][0]-ppa_list[1][0])**2)+ \
                             ((ppa_list[1][0]-ppa_list[1][1])**2) )

        print(distance)


    def update_table():
        pass

    #get the data from the sql database
    tableData = get_table_data()

    for idx, row in tableData.iterrows():

        if verbose:
            print("extract ppa for " + row["photo_id"] + "..", end='\r')

        if debug_show_ppa:
            img = load_image(row["file_path"])
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


        ppa_list = []
        #two methods for calculating ppa: with the side points and corner points
        if row["fid_mark_1_x"]is not None and not math.isnan(row["fid_mark_1_x"]): #corner

            #get the required points
            points = [[row["fid_mark_1_x"], row["fid_mark_1_y"]],
                      [row["fid_mark_2_x"], row["fid_mark_2_y"]],
                      [row["fid_mark_3_x"], row["fid_mark_3_y"]],
                      [row["fid_mark_4_x"], row["fid_mark_4_y"]]]

            ppa = calculate_ppa(points)
            ppa_list.append(ppa)

        if row["fid_mark_5_x"]is not None and not math.isnan(row["fid_mark_5_x"]): #sides

            #get the required points
            points = [[row["fid_mark_5_x"], row["fid_mark_5_y"]],
                      [row["fid_mark_6_x"], row["fid_mark_6_y"]],
                      [row["fid_mark_7_x"], row["fid_mark_7_y"]],
                      [row["fid_mark_8_x"], row["fid_mark_8_y"]]]

            ppa = calculate_ppa(points)
            ppa_list.append(ppa)

        if len(ppa_list) == 2:
            compare_ppas(ppa_list)

if __name__ == "__main__":

    extract_ppa()
