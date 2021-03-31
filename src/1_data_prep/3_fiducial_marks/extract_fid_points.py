"""
This algorithm calculate the coordinates of the fiducial points in the mid
of the image
"""

#debug params
verbose=True
debug_show_subsets = False
debug_show_points = False

import sys #in order to import other python files
import cv2 #in order for some cv methods
import numpy as np #for various stuff
import math #to check for nan
from scipy.signal import find_peaks, peak_widths # to find peaks

import matplotlib.pyplot as plt

#add for import
sys.path.append('../../misc')

from database_connections import *
from image_loader import *

def extract_fid_points():

    conn = Connector()

    def get_table_data():

        if verbose:
            print("get data from database..", end='\r')

        tables = ["image_properties", "images"]
        fields = [["photo_id",
                   "subset_width", "subset_height",
                   "subset_n_x", "subset_n_y",
                   "subset_e_x", "subset_e_y",
                   "subset_s_x", "subset_s_y",
                   "subset_w_x", "subset_w_y",
                  ],
                  ["file_path"]
                  ]
        join = ["photo_id"]
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
        tableData = conn.get_data(tables, fields, filters, join)

        if verbose:
            print("get data from database.. - finished")

        return tableData

    def load_subsets(img, imgParams):
        subsets = {"N":None,"E":None,"S":None,"W":None}
        subset_coordinates = {"N":None,"E":None,"S":None,"W":None}

        subset_width = imgParams["subset_width"]
        subset_height = imgParams["subset_height"]


        for key in subsets:
            if imgParams["subset_" + key.lower() + "_x"] is None or \
               math.isnan(imgParams["subset_" + key.lower() + "_x"]):
                continue
            if imgParams["subset_" + key.lower() + "_y"] is None or \
               math.isnan(imgParams["subset_" + key.lower() + "_y"]):
                continue

            x1 = int(imgParams["subset_" + key.lower() + "_x"])
            y1 = int(imgParams["subset_" + key.lower() + "_y"])

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            x2 = int(x1 + subset_width)
            y2 = int(y1 + subset_height)

            #for subsets in E the points are unusual far away, so here the subsets
            #must be larger
            if key == "E":
                x2 = x2 + 100

            #the points are usually at the outer edge of the subsets, so making
            #the subsets smaller helps finding them
            if key == "N":
                x1 = x1 + 50
                y2 = y2 - 150
                x2 = x2 - 50
            elif key == "E":
                y1 = y1 + 50
                x1 = x1 + 250
                y2 = y2 - 50
            elif key == "S":
                x1 = x1 + 50
                y1 = y1 + 150
                x2 = x2 - 50
            elif key == "W":
                y1 = y1 + 50
                x2 = x2 - 150
                y2 = y2 - 50

            subsets[key] = img[y1:y2,x1:x2]
            subset_coordinates[key] = [x1, y1, x2, y2]

        return subsets, subset_coordinates

    def show_subsets(imgId, subsets):
        f, axarr = plt.subplots(2,2)
        f.suptitle(imgId)
        f.tight_layout()
        if subsets["N"] is not None:
            axarr[0,0].imshow(subsets["N"], cmap='gray')
            axarr[0,0].set_title("N")
        if subsets["E"] is not None:
            axarr[0,1].imshow(subsets["E"], cmap='gray')
            axarr[0,1].set_title("E")
        if subsets["S"] is not None:
            axarr[1,0].imshow(subsets["S"], cmap='gray')
            axarr[1,0].set_title("S")
        if subsets["W"] is not None:
            axarr[1,1].imshow(subsets["W"], cmap='gray')
            axarr[1,1].set_title("W")
        plt.show()

    def binarize(subsets):

        binarized = {"N":None,"E":None,"S":None,"W":None}
        blurred = {"N":None,"E":None,"S":None,"W":None}
        for key in binarized:

            if subsets[key] is None:
                continue

            subset_blurred = cv2.GaussianBlur(subsets[key],(3,3),0)
            blurred[key] = subset_blurred

            #threshold
            ret,fid_th = cv2.threshold(subset_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            binarized[key] = fid_th

        return binarized

    def find_points(binarized, subsets):

        points = {"N":None,"E":None,"S":None,"W":None}
        for key in points:

            if binarized[key] is None:
                continue

            #find contours
            contours, hierarchy = cv2.findContours(binarized[key],
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            #get mid of image for late filtering
            midX, midY = int(subsets[key].shape[1])/2, int(subsets[key].shape[0]/2)

            #save only contours smaller than a threshold
            point_contours = []
            for contour in contours:
                #calculate area of point
                con_area = cv2.contourArea(contour)
                if 15 < con_area < 75:

                    #find the center of this area
                    M = cv2.moments(contour)

                    #get coordinates
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    point_contours.append((cX, cY))

            #too many points -> filter for the most suitable
            if len(point_contours) > 1:

                #save the point with smallest distance to middle
                minDistance = 100000
                minPoint = None

                #check distance and save the min distance
                for elem in point_contours:
                    distance = math.sqrt( ((midX-elem[0])**2)+((midY-elem[1])**2) )
                    if distance < minDistance:
                        minDistance = distance
                        minPoint = elem
                point_contours = [minPoint]

            #save point
            if len(point_contours) > 0:
                points[key] = [point_contours[0][0], point_contours[0][1]]

        return points

    def reproject_points(points, subset_coordinates):

        for key in points:

            point = points[key]
            s_coords = subset_coordinates[key]

            if point is None or s_coords is None:
                continue

            point[0] = point[0] + s_coords[0]
            point[1] = point[1] + s_coords[1]

            points[key] = point

        return points

    def show_points(img, points):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for key in points:
            point = points[key]
            if point is None:
                continue
            cv2.circle(img_rgb, (point[0], point[1]), 7, (255, 0, 0), 3)

        plt.imshow(img_rgb)
        plt.show()

    def update_table(conn, photo_id, points):

        table="image_properties"
        id={"photo_id":photo_id}
        data = {}

        for key in points:
            if points[key] is None:
                continue

            if key == "N":
                fid_pos = "7"
            elif key == "E":
                fid_pos = "6"
            elif key == "S":
                fid_pos = "8"
            elif key == "W":
                fid_pos = "5"

            data["fid_mark_" + fid_pos + "_x"] = points[key][0]
            data["fid_mark_" + fid_pos + "_y"] = points[key][1]

        data["point_extraction_date"] = "DATE()"

        conn.edit_data(table, id, data)

    #get the data from the sql database
    tableData = get_table_data()

    for idx, row in tableData.iterrows():

        if verbose:
            print("extracing points for " + row["photo_id"] + "..", end='\r')

        #load the images
        img = load_image(row['file_path'])

        subsets, subset_coordinates = load_subsets(img, row)

        if debug_show_subsets:
            show_subsets(row['photo_id'], subsets)

        binarized = binarize(subsets)

        points = find_points(binarized, subsets)

        points = reproject_points(points, subset_coordinates)

        if debug_show_points:
            show_points(img, points)

        update_table(conn, row["photo_id"], points)

        if verbose:
            print("extracing points for " + row["photo_id"] + ".. - finished")


if __name__ == "__main__":

    extract_fid_points()
