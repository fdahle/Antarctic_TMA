"""
This algorithm calculate the corner coordinates of an image
"""

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

#debug params
verbose=True
debug_show_subsets=False
debug_show_lines=False


def extract_corners():

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
                y2 = 0

            x2 = int(x1 + subset_width)
            y2 = int(y1 + subset_height)

            #for corner detection it is good to squeeze the subsets a little:
            #make them larger in the direction of lines and smaller in the
            #direction to the middle
            if key in ["N","S"]:
                x1 = x1 - 100
                x2 = x2 + 100
                y1 = y1 + 50
                y2 = y2 - 50
            elif key in ["E","W"]:
                x1 = x1 + 50
                x2 = x2 - 50
                y1 = y1 - 100
                y2 = y2 + 100

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

    def equalize(subsets):

        equalized = {"N":None,"E":None,"S":None,"W":None}
        for key in equalized:

            if subsets[key] is None:
                continue

            equalized[key] = cv2.equalizeHist(subsets[key])

        return equalized

    def binarize(subsets, equalized):

        binarized = {"N":None,"E":None,"S":None,"W":None}
        for key in binarized:

            if subsets[key] is None:
                continue

            #get histogramm and count number of peaks
            hst = cv2.calcHist([subsets[key]],[0],None,[256],[0,256])
            hst = np.ravel(hst)
            peaks, peak_props = find_peaks(hst, height=100, distance=10)
            nrOfHist = peaks.shape[0]

            if nrOfHist == 1:
                #subset = cv.medianBlur(subset,9,0)
                subset_blurred = cv2.GaussianBlur(equalized[key],(15,15),0)
            else:
                subset_blurred = cv2.GaussianBlur(subsets[key],(15,15),0)

            #threshold
            ret,fid_th = cv2.threshold(subset_blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            #smooth the threshold
            kernel = np.ones((15,15),np.uint8)
            binarized[key] = cv2.morphologyEx(fid_th, cv2.MORPH_OPEN, kernel)

        return binarized

    def find_edges(binarized):
        edges = {"N":None,"E":None,"S":None,"W":None}
        for key in edges:
            edges[key] = cv2.Canny(binarized[key], 1, 3)

        return edges

    def find_lines(edges):

        linesDict = {"N":None,"E":None,"S":None,"W":None}
        lines_good_quality = {"N":None,"E":None,"S":None,"W":None}
        #line settings
        rho = 1  # distance resolution in pixels of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 25  # minimum number of pixels making up a line
        max_line_gap = 25  # maximum gap in pixels between connectable line segments
        theta = np.pi/4 # the denominator tells in which steps is looked for lines (180 equals 1 degree)


        for key in linesDict:

            if edges[key] is None:
                continue

            #get mid of image
            mid_of_image_x= int(edges[key].shape[1]/2)
            mid_of_image_y = int(edges[key].shape[0]/2)

            #extract all lines
            allLines = cv2.HoughLinesP(edges[key], rho, theta, threshold, np.array([]),
                                        min_line_length, max_line_gap) #line is returned as [x0, y0, x1, y1]

            correct_lines=[]
            #iterate all the lines
            for line in allLines:

                #get slope of line
                with np.errstate(divide='ignore'):
                    line = line[0]
                    slope  = (line[3] - line[1]) / (line[2] - line[0])

                if key in ["N", "S"]:
                    if slope == 0:
                        correct_lines.append(line)

                if key in ["E", "W"]:
                    if math.isinf(slope):
                        correct_lines.append(line)

            #extract the line that is the closest to the middle. This is done twice
            # - once for top or left of the fiducial mark (1)
            # - once for the bottom or right of the fiducial mark(2)
            # - furthermore lines at the edges are excluded (150px currently)

            mid_line_1 = None
            mid_line_2 = None

            #minimum distance to the mid yet
            min_distance_1 = 150
            min_distance_2 = 150

            for line in correct_lines:

                #calculate distance
                if key in ["N", "S"]:
                    distance = np.abs(mid_of_image_y - line[1])
                elif key in ["E", "W"]:
                    distance = np.abs(mid_of_image_x - line[0])

                #get the closest lines
                if key in ["N", "S"]:
                    if line[0] < mid_of_image_x and distance < min_distance_1:
                        min_distance_1 = distance
                        mid_line_1 = line
                    if line[0] > mid_of_image_x and distance < min_distance_2:
                        min_distance_2 = distance
                        mid_line_2 = line
                elif key in ["E", "W"]:
                    if line[1] < mid_of_image_y and distance < min_distance_1:
                        min_distance_1 = distance
                        mid_line_1 = line
                    if line[1] > mid_of_image_y and distance < min_distance_2:
                        min_distance_2 = distance
                        mid_line_2 = line

            #check if lines are almost identical
            if mid_line_1 is not None and mid_line_2 is not None:
                if key in ["N", "S"]:
                    diff = np.abs(mid_line_1[1] - mid_line_2[1])
                elif key in ["E", "W"]:
                    diff = np.abs(mid_line_1[0] - mid_line_2[0])

                if diff > 10:
                    lines_good_quality[key] = False
                    if min_distance_1 < min_distance_2:
                        mid_line_2 = None
                    else:
                        mid_line_1 = None
                else:
                    lines_good_quality[key] = True

            #check how many lines could be found
            if mid_line_1 is None and mid_line_2 is None:
                continue
            elif mid_line_1 is None and mid_line_2 is not None:
                mid_line_1 = mid_line_2
            elif mid_line_1 is not None and mid_line_2 is None:
                mid_line_2 = mid_line_1

            #calculate a global line by using the innermost corners of each line
            #afterwards line is [x1, y1, x2 ,y2]
            if key in ["N", "S"]:
                line = [mid_line_1[2], mid_line_1[3], mid_line_2[0], mid_line_2[1]]
            elif key in ["E", "W"]:
                line = [mid_line_1[0], mid_line_1[1], mid_line_2[2], mid_line_2[3]]
            linesDict[key] = line

        return linesDict, lines_good_quality

    def reproject_lines(linesDict, subset_coordinates):
        #reproject the lines to the total coordinates
        for key in linesDict:

            if linesDict[key] is None or subset_coordinates is None:
                continue

            coords = linesDict[key]
            s_coords = subset_coordinates[key]

            coords[0] = coords[0] + s_coords[0]
            coords[2] = coords[2] + s_coords[0]
            coords[1] = coords[1] + s_coords[1]
            coords[3] = coords[3] + s_coords[1]

            linesDict[key] = coords

        return linesDict

    def show_lines(imgId, subsets, lines):
        f, axarr = plt.subplots(2,2)
        f.suptitle(imgId)
        f.tight_layout()

        if subsets["N"] is not None and lines["N"] is not None:
            subset_N = cv2.cvtColor(subsets["N"], cv2.COLOR_GRAY2BGR)
            cv2.line(subset_N, (lines["N"][0], lines["N"][1]),
                      (lines["N"][2], lines["N"][3]), (255, 255, 0), 2)
            axarr[0,0].imshow(subset_N)
            axarr[0,0].set_title("N")

        if subsets["E"] is not None and lines["E"] is not None:
            subset_E = cv2.cvtColor(subsets["E"], cv2.COLOR_GRAY2BGR)
            cv2.line(subset_E, (lines["E"][0], lines["E"][1]),
                      (lines["E"][2], lines["E"][3]), (255, 255, 0), 2)
            axarr[0,1].imshow(subset_E)
            axarr[0,1].set_title("E")
        if subsets["S"] is not None and lines["S"] is not None:
            subset_S = cv2.cvtColor(subsets["S"], cv2.COLOR_GRAY2BGR)
            cv2.line(subset_S, (lines["S"][0], lines["S"][1]),
                      (lines["S"][2], lines["S"][3]), (255, 255, 0), 2)
            axarr[1,0].imshow(subset_S)
            axarr[1,0].set_title("S")
        if subsets["W"] is not None and lines["W"] is not None:
            subset_W = cv2.cvtColor(subsets["W"], cv2.COLOR_GRAY2BGR)
            cv2.line(subset_W, (lines["W"][0], lines["W"][1]),
                      (lines["W"][2], lines["W"][3]), (255, 255, 0), 2)
            axarr[1,1].imshow(subset_W)
            axarr[1,1].set_title("W")
        plt.show()

    def find_corners(lines):

        corners = {"NE":None,"SE":None,"SW":None,"NW":None}

        for key in corners:

            direction_1, direction_2 = key[0], key[1]

            line1 = lines[direction_1]
            line2 = lines[direction_2]

            #cannot calculate the corner, as one line is missing
            if line1 is None or line2 is None:
                continue

            x1=line1[0]
            x2=line1[2]
            y1=line1[1]
            y2=line1[3]
            x3=line2[0]
            x4=line2[2]
            y3=line2[1]
            y4=line2[3]

            #calcualte the point coordinates
            D = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/D
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/D

            px = int(px)
            py = int(py)

            corners[key]= (px, py)
        return corners

    def update_table(conn, photo_id, lines, lines_good_quality, corners):

        table="image_properties"
        id={"photo_id":photo_id}
        data = {}

        for key in lines:
            if lines[key] is None:
                continue

            data["line_" + key + "_x1"] = lines[key][0]
            data["line_" + key + "_y1"] = lines[key][1]
            data["line_" + key + "_x2"] = lines[key][2]
            data["line_" + key + "_y2"] = lines[key][3]

        for key in corners:

            if corners[key] is None:
                continue

            #get the right number of fiducial point
            fid_pos = None
            if key == "NE":
                fid_pos = "2"
            elif key == "SE":
                fid_pos = "4"
            elif key == "SW":
                fid_pos = "1"
            elif key == "NW":
                fid_pos = "3"

            data["fid_mark_" + fid_pos + "_x"] = corners[key][0]
            data["fid_mark_" + fid_pos + "_y"] = corners[key][1]

            if lines_good_quality[key[0]] is False or \
            lines_good_quality[key[1]] is False:
                data["fid_mark_" + fid_pos + "_estimated"] = "TRUE()"
            else:
                data["fid_mark_" + fid_pos + "_estimated"] = "FALSE()"

        data["line_extraction_date"] = "DATE()"

        conn.edit_data(table, id, data)

    #get the data from the sql database
    tableData = get_table_data()

    for idx, row in tableData.iterrows():

        if verbose:
            print("extracing corners for " + row["photo_id"] + "..", end='\r')

        #load the images
        img = load_image(row['file_path'])

        subsets, subset_coordinates = load_subsets(img, row)

        if subsets["N"] is None and subsets["E"] is None and \
        subsets["S"] is None and subsets["W"] is None:
            print("extracing corners for " + row["photo_id"] + ".. - no image data found")
            continue

        if debug_show_subsets:
            show_subsets(row['photo_id'], subsets)

        equalized = equalize(subsets)

        binarized = binarize(subsets, equalized)

        edges = find_edges(binarized)

        lines, lines_good_quality = find_lines(edges)

        if debug_show_lines:
            show_lines(row['photo_id'], subsets, lines)

        lines = reproject_lines(lines, subset_coordinates)

        corners = find_corners(lines)

        update_table(conn, row["photo_id"], lines, lines_good_quality, corners)

        if verbose:
            print("extracing corners for " + row["photo_id"] + ".. - finished")


if __name__ == "__main__":

    extract_corners()
