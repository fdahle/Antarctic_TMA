import json
import random

import cv2
import PIL.Image
import numpy as np
import os
import time
import copy
import csv

from tkinter import *
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import font
import webbrowser

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import colors
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches

import functions.remove_small_clusters as rsc
import functions.set_class as sc
from functions.segment import segment
from database_connection import Connector

with open('../params1.json') as json_file:
    params1 = json.load(json_file)

# if a path is specified only segments from this csv are taken
segment_csv_path = "/home/fdahle/Desktop/ATM/data/_need_to_check/segment.csv"

bool_no_finished = True

data_folder = "../../../data/"

fileFolder = data_folder + "aerial/TMA/downloaded"
segmentFolder = data_folder + "aerial/TMA/segmented/unsupervised"
outputFolder = data_folder + "aerial/TMA/segmented/supervised"

decrease_factor = params1["segmentator_decrease_factor"]
bool_improve_images = params1["segmentator_bool_improve_images"]  # remove small clusters from the image

minClusterSize = 25000

showAllFiles = params1["segmentator_bool_show_files_with_output"]
figsize = (4, 4)
#           1       2       3       4         5        6        7
classes = ["Ice", "Snow", "Rocks", "Water", "Clouds", "Sky", "Unknown"]
custom_colors = [(242 / 255, 170 / 255, 217 / 255), (150 / 255, 149 / 255, 158 / 255),
                 (230 / 255, 230 / 255, 235 / 255), (46 / 255, 45 / 255, 46 / 255),
                 (7 / 255, 29 / 255, 232 / 255), (25 / 255, 227 / 255, 224 / 255),
                 (186 / 255, 39 / 255, 32 / 255), (224 / 255, 7 / 255, 224 / 255)]
limits = range(0, 9)
cmapOutput, normOutput = from_levels_and_colors(limits, custom_colors)
random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))

edge_output = 25  # segmentation not working good at the edge
edge_total = 0


def quit_me():
    window.quit()
    window.destroy()
    exit()


# iterate the fileFolder to get all images and save them
filesList = []

if segment_csv_path == "":
    for filename in os.listdir(fileFolder):
        if filename.endswith(".tif"):

            if showAllFiles == "False" and os.path.isfile(outputFolder + "/" + filename):
                continue

            # show only files with an unsupervised files
            if os.path.isfile(segmentFolder + "/" + filename) is False:
                continue

            # don't show files already finished
            if os.path.isfile(outputFolder + "/" + filename):
                continue

            filesList.append(filename)
else:
    with open(segment_csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:

            # show only files with an unsupervised files
            if os.path.isfile(segmentFolder + "/" + row[0]) is False:
                continue

            # don't show files already finished
            if os.path.isfile(outputFolder + "/" + row[0]):
                continue

            filesList.append(row)

    filesList = [item for sublist in filesList for item in sublist]

window = Tk()
window.title('Segmentator')
window.geometry('1450x1000+10+10')
window.protocol("WM_DELETE_WINDOW", quit_me)

frame = Frame(window, relief='sunken')
frame.pack(fill=BOTH, expand=True, padx=10, pady=20)

# lock variable for hover and click
img_lock = False
img_loaded = False
bool_scribbled = False

bool_left_clicked = False
bool_crosshair = False

bool_boxDrawing = False
boxTopY = None
boxLeftX = None
boxBottomY = None
boxRightX = None

fileId = None
imgClustered = None
img_segmented_never_change = np.zeros([2,2])
img_segmented = np.zeros([2, 2])
img_segmentedOrig = np.zeros([2, 2])

origX, origY = None, None

labelLink = ""

bool_overwrite = IntVar()
bool_overwrite_same_class = IntVar()

scribble_size = 20

conn = Connector(catch=False)


def wait(message):
    win = Toplevel(window)
    win.transient()
    win.title('Please wait')
    Label(win, text=message).pack()
    return win


def get_random_img():
    random_img = random.choice(filesList)
    on_img_select(random_id=random_img)


def set_failed_img():

    global filesList

    os.rename(segmentFolder + "/" + str(fileId), segmentFolder + "/failed/" + str(fileId))
    filesList.remove(str(fileId))
    edit_status("Image is set to failed")


# action when an image is selected
def on_img_select(*args, random_id=None):
    start_time = time.time()

    win = wait('image is loading...')

    # restore box params
    global bool_boxDrawing
    global boxTopY
    global boxLeftX
    global boxBottomY
    global boxRightX

    bool_boxDrawing = False
    boxTopY = None
    boxLeftX = None
    boxBottomY = None
    boxRightX = None

    # remove possible boxes
    axSegmented.patches = []

    global img_lock
    global img_loaded
    global fileId

    # lock image for hover and click
    img_lock = True

    if random_id is None:
        fileId = optionsImg.get()
    else:
        fileId = random_id

    # get the file paths
    img_path = fileFolder + "/" + fileId
    segments_path = segmentFolder + "/" + fileId
    output_path = outputFolder + "/" + fileId

    # set the label
    global labelLink
    labelSelectedImg.config(text=fileId)
    labelLink = os.path.abspath(img_path)

    global imgRaw
    global imgOutput
    global img_segmented
    global img_segmentedOrig
    global img_segmented_never_change

    global origX
    global origY

    def cut_off_edge(img, file_id):

        short_id = "'" + file_id[:-4] + "'"

        sql_string = "SELECT fid_mark_1_x, " + \
                     "fid_mark_1_y, " + \
                     "fid_mark_2_x, " + \
                     "fid_mark_2_y, " + \
                     "fid_mark_3_x, " + \
                     "fid_mark_3_y, " + \
                     "fid_mark_4_x, " + \
                     "fid_mark_4_y " + \
                     "FROM images_properties " + \
                     "WHERE image_id=" + short_id

        # get data from table
        table_data = conn.get_data(sql_string)

        subset_border = params1["unsupervised_subset_border"]

        # get left
        if table_data["fid_mark_1_x"].item() >= table_data["fid_mark_3_x"].item():
            left = table_data["fid_mark_1_x"].item()
        else:
            left = table_data["fid_mark_3_x"].item()

        # get top
        if table_data["fid_mark_2_y"].item() >= table_data["fid_mark_3_y"].item():
            top = table_data["fid_mark_2_y"].item()
        else:
            top = table_data["fid_mark_3_y"].item()

        # get right
        if table_data["fid_mark_2_x"].item() <= table_data["fid_mark_4_x"].item():
            right = table_data["fid_mark_2_x"].item()
        else:
            right = table_data["fid_mark_4_x"].item()

        # get bottom
        if table_data["fid_mark_1_y"].item() <= table_data["fid_mark_4_y"].item():
            bottom = table_data["fid_mark_1_y"].item()
        else:
            bottom = table_data["fid_mark_4_y"].item()

        left = int(left + subset_border)
        right = int(right - subset_border)
        top = int(top + subset_border)
        bottom = int(bottom - subset_border)

        img = img[top:bottom, left:right]

        return img

    edit_status("Step 1 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    # load images as np arrays
    img_segmented = np.array(PIL.Image.open(segments_path))
    img_segmented = img_segmented[edge_output:img_segmented.shape[0] - edge_output,
                                  edge_output:img_segmented.shape[1] - edge_output]
    img_segmented_never_change = copy.deepcopy(img_segmented)

    edit_status("Step 2 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    imgRaw = np.array(PIL.Image.open(img_path))
    imgRaw = cut_off_edge(imgRaw, fileId)
    imgRaw = imgRaw[edge_output:imgRaw.shape[0] - edge_output, edge_output:imgRaw.shape[1] - edge_output]

    edit_status("Step 3 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    origY, origX = imgRaw.shape[0], imgRaw.shape[1]

    if decrease_factor > 0:
        new_y = int(img_segmented.shape[0] / decrease_factor)
        new_x = int(img_segmented.shape[1] / decrease_factor)
        img_segmented = cv2.resize(img_segmented, dsize=(new_y, new_x), interpolation=cv2.INTER_NEAREST)
        imgRaw = cv2.resize(imgRaw, dsize=(new_y, new_x), interpolation=cv2.INTER_NEAREST)

    try:
        imgOutput = np.array(PIL.Image.open(output_path))
        imgOutput = imgOutput[edge_total:imgOutput.shape[0] - edge_total, edge_total:imgOutput.shape[1] - edge_total]
        imgOutput = cv2.resize(imgOutput, dsize=(img_segmented.shape[0], img_segmented.shape[1]),
                               interpolation=cv2.INTER_NEAREST)
    except (Exception,):
        imgOutput = np.zeros_like(img_segmented)

    edit_status("Step 4 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    if bool_improve_images == "True":
        img_segmented = rsc.remove_small_clusters(img_segmented, minClusterSize)

    edit_status("Step 5 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    # copy for scribbles
    img_segmentedOrig = copy.deepcopy(img_segmented)

    # display images
    axRaw.imshow(imgRaw, cmap='gray')
    figRaw.canvas.draw_idle()
    axSegmented.imshow(img_segmented, cmap=random_cmap, interpolation='None')
    figSegmented.canvas.draw_idle()
    axOutput.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none')
    figOutput.canvas.draw_idle()
    axOverlay.imshow(imgRaw, cmap="gray")
    axOverlay.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none', alpha=0.5)
    figOverlay.canvas.draw_idle()

    edit_status("Step 6 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    # load segment tree
    unique, counts = np.unique(img_segmented, return_counts=True)
    class_dict = dict(zip(unique, counts))

    edit_status("Step 7 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    # fill the legend of the tree
    treeSegments.delete(*treeSegments.get_children())
    for elem in class_dict:
        treeSegments.insert("", "end", values=(elem, class_dict[elem]))

    edit_status("Step 8 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    # refresh the classified tree
    refresh_tree()

    # reset locking of images
    img_lock = False
    img_loaded = True

    win.destroy()

    edit_status("Step 9 done in " + str(time.time() - start_time) + " seconds")
    edit_status("image is loaded")


# action when clicked on a image
def on_img_click(event):
    if img_lock:
        return

    if img_loaded is False:
        return

    if event.xdata is None or event.ydata is None:
        return

    global bool_left_clicked
    bool_left_clicked = True

    global bool_boxDrawing
    global boxTopY
    global boxLeftX
    global boxBottomY
    global boxRightX

    x, y = int(event.xdata), int(event.ydata)

    # if we draw scribbles or boxes no need to update the textBoxes with x or y
    if bool_crosshair:

        # if we are drawing a box we need to get the coordinates
        if bool_boxDrawing:

            if boxTopY is None and boxLeftX is None:
                boxTopY = y
                boxLeftX = x
                edit_status("Draw Box: TopY:{} LeftX:{}".format(y, x))
            else:
                boxBottomY = y
                boxRightX = x
                edit_status("Draw Box: BottomY:{} RightX:{}".format(y, x))

                if boxTopY >= boxBottomY or boxLeftX >= boxRightX:
                    edit_status("Draw Box: Something went wrong. Please Draw the box again")
                else:

                    height = boxBottomY - boxTopY
                    width = boxRightX - boxLeftX

                    rect = patches.Rectangle((boxLeftX, boxTopY), width, height, linewidth=1, edgecolor='r',
                                             facecolor='none')
                    axSegmented.add_patch(rect)
                    figSegmented.canvas.draw_idle()

                bool_boxDrawing = False
        return

    textX.delete(1.0, "end-1c")
    textX.insert("end-1c", x)
    textY.delete(1.0, "end-1c")
    textY.insert("end-1c", y)


def on_mouse_release(event):
    global bool_left_clicked
    global bool_crosshair

    bool_left_clicked = False

    if bool_crosshair:

        if bool_boxDrawing:
            return

        axSegmented.imshow(img_segmented, cmap=random_cmap, interpolation='None')
        figSegmented.canvas.draw_idle()

        window.config(cursor="arrow")
        bool_crosshair = False


# action when hovering over the image
def on_img_hover(event):
    global bool_scribbled

    if img_lock:
        return

    if img_loaded is False:
        return

    if event.xdata is None or event.ydata is None:
        varX.set("x: ...")
        varY.set("y: ...")
        varClass.set("class: ...")
        return

    x, y = int(event.xdata), int(event.ydata)
    varX.set("x: " + str(x))
    varY.set("y: " + str(y))
    varClass.set("class: " + str(img_segmented[y, x]))

    final_id = imgOutput[y, x]
    if final_id == 0:
        final_class = "not set"
    else:
        final_class = classes[final_id - 1]
    varFinalClass.set("final class: " + final_class)

    if bool_left_clicked and bool_crosshair and bool_boxDrawing is False:

        min_x = x - scribble_size
        if min_x < 0:
            min_x = 0

        min_y = y - scribble_size
        if min_y < 0:
            min_y = 0

        max_x = x + scribble_size
        if max_x > img_segmented.shape[1]:
            max_x = img_segmented.shape[1]

        max_y = y + scribble_size
        if max_y > img_segmented.shape[0]:
            max_y = img_segmented.shape[0]

        img_segmented[min_y:max_y, min_x:max_x] = 100
        bool_scribbled = True


def on_button_scribble():

    global scribble_size

    try:
        new_scribble_size = int(textScribbleSize.get())
    except (Exception,):
        edit_status("Scribble: Please enter a positive value")
        return

    scribble_size = np.abs(new_scribble_size)

    if img_lock:
        return

    if img_loaded is False:
        edit_status("Scribble: To use this function please load an image")
        return

    window.config(cursor="crosshair")
    global bool_crosshair
    bool_crosshair = True


def on_button_clear():
    global bool_scribbled
    global img_segmented

    if img_lock:
        return

    if img_loaded is False:
        edit_status("Clear: To use this function please load an image")
        return

    if bool_scribbled:
        img_segmented = copy.deepcopy(img_segmentedOrig)
        axSegmented.imshow(img_segmented, cmap=random_cmap, interpolation='None')
        figSegmented.canvas.draw_idle()
        bool_scribbled = False


def on_button_set_class():
    start_time = time.time()

    global imgOutput
    global img_segmented
    global imgClustered

    global bool_scribbled

    if img_lock:
        return

    if img_loaded is False:
        edit_status("Set Class: To use this function please load an image")
        return

    x = textX.get("1.0", END)
    y = textY.get("1.0", END)

    # check if it is a valid number
    try:
        x = int(x)
        y = int(y)
    except (Exception,):
        edit_status("X or Y is not a valid number")
        return

    # check if numbers in range
    if x < 0 or x > img_segmented.shape[0]:
        edit_status("X is not a valid value")
        return
    elif y < 0 or y > img_segmented.shape[1]:
        edit_status("Y is not a valid value")
        return

    # get segmentId
    if comboClass.get() == '':
        edit_status("Please select a class")
        return

    edit_status("1--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    segment_id = classes.index(comboClass.get()) + 1

    imgOutput, imgClustered = sc.set_class(img_segmented, imgOutput, x, y, segment_id, imgClustered, bool_scribbled,
                                           bool_overwrite.get(), bool_overwrite_same_class)

    edit_status("2--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    axOutput.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none')
    figOutput.canvas.draw_idle()
    axOverlay.imshow(imgRaw, cmap="gray")
    axOverlay.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none', alpha=0.5)
    figOverlay.canvas.draw_idle()

    edit_status("3--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # remove scribbles if available
    if bool_scribbled:
        img_segmented = img_segmentedOrig.copy()
        axSegmented.imshow(img_segmented, cmap=random_cmap, interpolation='None')
        figSegmented.canvas.draw_idle()
        bool_scribbled = False

    edit_status("4--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # refresh the tree
    refresh_tree()

    edit_status("5--- %s seconds ---" % (time.time() - start_time))

    # reset the x,y,class elements
    textX.delete(1.0, "end")
    textY.delete(1.0, "end")

    edit_status("Class is set")


def on_button_set_all():
    global imgOutput
    global bool_scribbled
    global img_segmented
    
    if img_lock:
        return

    if img_loaded is False:
        edit_status("Set All: To use this function please load an image")
        return

    # get segmentId
    if comboClass.get() == '':
        edit_status("Please select a class")
        return

    msg_box = messagebox.askquestion('Set all classes',
                                     'Are you sure you want to set all classes?', icon='warning')
    if msg_box == 'yes':
        pass  # do nothing and continue
    else:
        return

    segment_id = classes.index(comboClass.get()) + 1

    # remove scribbles if available
    if bool_scribbled:
        img_segmented = copy.deepcopy(img_segmentedOrig)
        axSegmented.imshow(img_segmented, cmap=random_cmap, interpolation='None')
        figSegmented.canvas.draw_idle()
        bool_scribbled = False

    imgOutput[imgOutput == 0] = segment_id

    # refresh the tree
    refresh_tree()

    axOutput.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none')
    figOutput.canvas.draw_idle()
    axOverlay.imshow(imgRaw, cmap="gray")
    axOverlay.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none', alpha=0.5)
    figOverlay.canvas.draw_idle()


def on_button_save():
    global imgOutput

    if img_lock:
        return

    if img_loaded is False:
        edit_status("Save: To use this function please load an image")
        return

    if fileId is None:
        return

    # replace not set with unknown
    imgOutput[imgOutput == 0] = 7
    imgOutput[imgOutput > 100] = 7

    if decrease_factor > 0:
        imgOutput = cv2.resize(imgOutput, dsize=(origX, origY), interpolation=cv2.INTER_NEAREST)

    # increase image size so that segmentation fit's to the normal image
    imgOutput = np.pad(imgOutput, pad_width=edge_output, mode='constant', constant_values=7)



    cv2.imwrite(outputFolder + "/" + str(fileId), imgOutput)

    global filesList
    os.rename(segmentFolder + "/" + str(fileId), segmentFolder + "/done/" + str(fileId))
    filesList.remove(str(fileId))

    edit_status("Image is saved")


def on_button_reset():
    global imgOutput

    if img_lock:
        return

    if img_loaded is False:
        edit_status("Reset: To use this function please load an image")
        return

    msg_box = messagebox.askquestion('Reset image',
                                     'Are you sure you want to reset the image?', icon='warning')
    if msg_box == 'yes':
        pass  # do nothing and continue
    else:
        return

    # set everything to zero
    imgOutput = np.zeros_like(img_segmented)

    # refresh the tree
    refresh_tree()

    axOutput.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none')
    figOutput.canvas.draw_idle()
    axOverlay.imshow(imgRaw, cmap="gray")
    axOverlay.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none', alpha=0.5)
    figOverlay.canvas.draw_idle()


def on_button_draw_box():
    if img_lock:
        return

    if img_loaded is False:
        edit_status("Draw Box: To use this function please load an image")
        return

    global boxTopY
    global boxBottomY
    global boxLeftX
    global boxRightX

    # if a box is already there delete the old box
    if boxTopY is not None and boxBottomY is not None and boxLeftX is not None and boxRightX is not None:
        boxTopY = -1
        boxBottomY = -1
        boxLeftX = -1
        boxRightX = -1
        axSegmented.patches = []
        figSegmented.canvas.draw_idle()

    edit_status("Draw Box: Please click first topLeft and then Bottom-Right")

    window.config(cursor="crosshair")
    global bool_crosshair
    bool_crosshair = True

    global bool_boxDrawing
    bool_boxDrawing = True


def on_button_delete_box():
    if img_lock:
        return

    if img_loaded is False:
        edit_status("Delete Box: To use this function please load an image")
        return

    global boxTopY
    global boxBottomY
    global boxLeftX
    global boxRightX

    boxTopY = None
    boxBottomY = None
    boxLeftX = None
    boxRightX = None

    axSegmented.patches = []
    figSegmented.canvas.draw_idle()

    global bool_boxDrawing
    bool_boxDrawing = False

    global bool_crosshair
    window.config(cursor="arrow")
    bool_crosshair = False


def on_button_cluster():
    if img_lock:
        return

    if img_loaded is False:
        edit_status("Cluster: To use this function please load an image")
        return

    nr_clusters = TextNumClasses.get("1.0", END)

    try:
        nr_clusters = int(nr_clusters)
    except (Exception,):
        edit_status("Cluster: Nr. of Clusters is not a valid number")
        return

    if nr_clusters < 0:
        edit_status("Cluster: Nr. of Clusters is not a valid number")
        return

    global img_segmented
    global imgOutput

    global boxTopY
    global boxBottomY
    global boxLeftX
    global boxRightX

    if boxTopY is None or boxBottomY is None or boxLeftX is None or boxRightX is None:
        edit_status("Cluster: Something went wrong. Please redraw the box")
        return

    subset = imgRaw[boxTopY:boxBottomY, boxLeftX:boxRightX]
    old_segmented_subset = img_segmented[boxTopY:boxBottomY, boxLeftX:boxRightX]
    class_subset = imgOutput[boxTopY:boxBottomY, boxLeftX:boxRightX]

    max_segment_id = np.amax(img_segmented)

    edit_status("Cluster: Reclustering begins")
    segmented_subset = segment(subset, nr_clusters, max_segment_id)
    edit_status("Cluster: Reclustering finished")

    # replace values
    segmented_subset[class_subset != 0] = old_segmented_subset[class_subset != 0]

    # make segmentedSubset smaller, as the edge always sucks
    edge = 10
    segmented_subset = segmented_subset[edge:segmented_subset.shape[0] - edge, edge:segmented_subset.shape[1] - edge]

    # replace the drawing with newly segmented
    img_segmented[boxTopY + edge:boxBottomY - edge, boxLeftX + edge:boxRightX - edge] = segmented_subset

    axSegmented.imshow(img_segmented, cmap=random_cmap, interpolation='None')
    figSegmented.canvas.draw_idle()
    axOutput.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none')
    figOutput.canvas.draw_idle()
    axOverlay.imshow(imgRaw, cmap="gray")
    axOverlay.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none', alpha=0.5)
    figOverlay.canvas.draw_idle()

    boxTopY = -1
    boxBottomY = -1
    boxLeftX = -1
    boxRightX = -1

    axSegmented.patches = []
    figSegmented.canvas.draw_idle()

    global bool_boxDrawing
    bool_boxDrawing = False


# with watershed min cluster
def on_button_recluster():

    global img_segmented
    global img_segmentedOrig
    global img_lock
    global img_loaded
    global minClusterSize

    img_lock = True
    img_loaded = False

    start_time = time.time()

    try:
        minClusterSize = int(textMinClusterSize.get())
    except (Exception,):
        print("please enter positive int values")
        return

    img_segmented = rsc.remove_small_clusters(img_segmented_never_change, minClusterSize)

    edit_status("reclustering done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    # copy for scribbles
    img_segmentedOrig = copy.deepcopy(img_segmented)

    # display images
    axRaw.imshow(imgRaw, cmap='gray')
    figRaw.canvas.draw_idle()
    axSegmented.imshow(img_segmented, cmap=random_cmap, interpolation='None')
    figSegmented.canvas.draw_idle()
    axOutput.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none')
    figOutput.canvas.draw_idle()
    axOverlay.imshow(imgRaw, cmap="gray")
    axOverlay.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none', alpha=0.5)
    figOverlay.canvas.draw_idle()

    edit_status("Step 6 done in " + str(time.time() - start_time) + " seconds")
    start_time = time.time()

    # load segment tree
    unique, counts = np.unique(img_segmented, return_counts=True)
    class_dict = dict(zip(unique, counts))

    # fill the legend of the tree
    treeSegments.delete(*treeSegments.get_children())
    for elem in class_dict:
        treeSegments.insert("", "end", values=(elem, class_dict[elem]))

    # refresh the classified tree
    refresh_tree()

    # reset locking of images
    img_lock = False
    img_loaded = True


# add something to the status field
def edit_status(text):
    textStatus.configure(state='normal')
    textStatus.insert('end', text + "\n")
    textStatus.configure(state='disabled')
    textStatus.see("end")


# refresh the tree with the classes
def refresh_tree():
    # fill the legend of output
    unique, counts = np.unique(imgOutput, return_counts=True)
    class_dict = dict(zip(unique, counts))

    # fill the legend of the tree
    treeOutput.delete(*treeOutput.get_children())

    if 0 in class_dict:
        treeOutput.insert("", "end", values=("not set", class_dict[0]))

    for i, elem in enumerate(classes):
        j = i + 1
        if j in class_dict:
            treeOutput.insert("", "end", values=(elem, class_dict[j]))


# image selector
optionsImg = StringVar(frame)
optionsImg.trace("w", on_img_select)
imageSelect = OptionMenu(frame, optionsImg, *filesList)

# random image selector
buttonFailImg = Button(frame, text="Failed", command=set_failed_img)
buttonRandomImg = Button(frame, text="Random", command=get_random_img)

# link to the image
labelSelectedImg = Label(frame, fg="blue", cursor="hand2")
f = font.Font(labelSelectedImg, labelSelectedImg.cget("font"))
f.configure(underline=True)
labelSelectedImg.configure(font=f)
labelSelectedImg.bind("<Button-1>", lambda e: webbrowser.open_new(r"file://" + labelLink))

varMinClusterSize = StringVar()
varMinClusterSize.set("minClusterSize")
labelMinClusterSize = Label(frame, textvariable=varMinClusterSize)

textMinClusterSize = Entry(frame, width=6)
textMinClusterSize.insert(0, minClusterSize)
buttonMinClusterSize = Button(frame, text="Cluster", command=on_button_recluster)


# x and y position
varX = StringVar()
varX.set("x: ...")
labelX = Label(frame, textvariable=varX)

varY = StringVar()
varY.set("y: ...")
labelY = Label(frame, textvariable=varY)

varClass = StringVar()
varClass.set("class: ...")
labelClass = Label(frame, textvariable=varClass)

varFinalClass = StringVar()
varFinalClass.set("final class: ...")
labelFinalClass = Label(frame, textvariable=varFinalClass)

# draw the segmented image
varSegmented = StringVar()
varSegmented.set("segmented image")
labelSegmented = Label(frame, textvariable=varSegmented)

figSegmented = Figure(figsize=figsize)
axSegmented = figSegmented.add_subplot(111)
axSegmented.imshow(img_segmented, cmap=random_cmap)

# remove whitespace around image
figSegmented.subplots_adjust(left=0, right=1, bottom=0, top=1)
axSegmented.set_axis_off()

# add events and draw
canvasSegmented = FigureCanvasTkAgg(figSegmented, master=frame)
canvasSegmented.draw()

figSegmented.canvas.mpl_connect('button_press_event', on_img_click)
figSegmented.canvas.mpl_connect('button_release_event', on_mouse_release)
figSegmented.canvas.mpl_connect("motion_notify_event", on_img_hover)

# draw the raw image
varRaw = StringVar()
varRaw.set("raw image")
labelRaw = Label(frame, textvariable=varRaw)

imgRaw = np.zeros((2, 2))
figRaw = Figure(figsize=figsize)
figRaw.subplots_adjust(left=0, right=1, bottom=0, top=1)
axRaw = figRaw.add_subplot(111)
axRaw.imshow(imgRaw, cmap="gray")

# remove whitespace around image
figRaw.subplots_adjust(left=0, right=1, bottom=0, top=1)
axRaw.set_axis_off()

canvasRaw = FigureCanvasTkAgg(figRaw, master=frame)
canvasRaw.draw()

figRaw.canvas.mpl_connect('button_press_event', on_img_click)
figRaw.canvas.mpl_connect('button_release_event', on_mouse_release)
figRaw.canvas.mpl_connect("motion_notify_event", on_img_hover)

# draw the output image
varOutput = StringVar()
varOutput.set("output image")
labelOutput = Label(frame, textvariable=varOutput)

imgOutput = np.zeros_like(img_segmented)
figOutput = Figure(figsize=figsize)
axOutput = figOutput.add_subplot(111)
axOutput.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none')

# remove whitespace around image
figOutput.subplots_adjust(left=0, right=1, bottom=0, top=1)
axOutput.set_axis_off()

canvasOutput = FigureCanvasTkAgg(figOutput, master=frame)
canvasOutput.draw()

figOutput.canvas.mpl_connect('button_press_event', on_img_click)
figOutput.canvas.mpl_connect('button_release_event', on_mouse_release)
figOutput.canvas.mpl_connect("motion_notify_event", on_img_hover)

# draw the overlay image
varOverlay = StringVar()
varOverlay.set("overlay image")
labelOverlay = Label(frame, textvariable=varOverlay)

figOverlay = Figure(figsize=figsize)
axOverlay = figOverlay.add_subplot(111)
axOverlay.imshow(imgRaw, cmap="gray")
axOverlay.imshow(imgOutput, cmap=cmapOutput, norm=normOutput, interpolation='none', alpha=0.5)

figOverlay.subplots_adjust(left=0, right=1, bottom=0, top=1)
axOverlay.set_axis_off()

canvasOverlay = FigureCanvasTkAgg(figOverlay, master=frame)
canvasOverlay.draw()

figOverlay.canvas.mpl_connect('button_press_event', on_img_click)
figOverlay.canvas.mpl_connect('button_release_event', on_mouse_release)
figOverlay.canvas.mpl_connect("motion_notify_event", on_img_hover)

# the tree with the segments
varTreeSegments = StringVar()
varTreeSegments.set("temporary segments")
labelTreeSegments = Label(frame, textvariable=varTreeSegments)
treeSegments = ttk.Treeview(frame, columns=('SegmentId', 'Count'), show='headings')

# the tree with the segments
varTreeOutputs = StringVar()
varTreeOutputs.set("final segments")
labelTreeOutputs = Label(frame, textvariable=varTreeOutputs)
treeOutput = ttk.Treeview(frame, columns=('SegmentId', 'Count'), show='headings')

checkOverwrite = Checkbutton(frame, text="overwrite", variable=bool_overwrite)
check_overwrite_only_same_class = Checkbutton(frame, text="only same class", variable=bool_overwrite_same_class)

buttonScribble = Button(frame, text="Scribble", command=on_button_scribble)
textScribbleSize = Entry(frame, width=3)
textScribbleSize.insert(0, "20")
buttonClear = Button(frame, text="Clear", command=on_button_clear)

# the input for x and y to change
varTextX = StringVar()
varTextX.set("x:")
labelTextX = Label(frame, textvariable=varTextX)
textX = Text(frame, height=1, width=5)

varTextY = StringVar()
varTextY.set("y:")
labelTextY = Label(frame, textvariable=varTextY)
textY = Text(frame, height=1, width=5)

optionsClasses = StringVar()
comboClass = ttk.Combobox(frame, width=27,
                          textvariable=optionsClasses)
comboClass['values'] = classes

buttonClass = Button(frame, text="Set class", command=on_button_set_class)
buttonSetAll = Button(frame, text="Set All", command=on_button_set_all)

buttonDrawBox = Button(frame, text="Draw Box", command=on_button_draw_box)
buttonDeleteBox = Button(frame, text="DeleteBox", command=on_button_delete_box)
buttonCluster = Button(frame, text="Cluster", command=on_button_cluster)
varTextNumClasses = StringVar()
varTextNumClasses.set("Classes:")
labelTextNumClasses = Label(frame, textvariable=varTextNumClasses)
TextNumClasses = Text(frame, height=1, width=3)

buttonSave = Button(frame, text="Save", command=on_button_save)
buttonReset = Button(frame, text="Reset", command=on_button_reset)

textStatus = scrolledtext.ScrolledText(frame, height=3, state='disabled')

# grid settings
buttonFailImg.grid(row=0, column=1)
imageSelect.grid(row=0, column=2, columnspan=3)
buttonRandomImg.grid(row=0, column=6)

labelSelectedImg.grid(row=0, column=4, columnspan=2)

labelX.grid(row=0, column=11)
labelY.grid(row=0, column=12)
labelClass.grid(row=0, column=14)
labelFinalClass.grid(row=0, column=15)


labelMinClusterSize.grid(row=1, column=1)
textMinClusterSize.grid(row=1, column=2)
buttonMinClusterSize.grid(row=1, column=3)


labelSegmented.grid(row=1, column=8, columnspan=5)
canvasSegmented.get_tk_widget().grid(row=2, column=8, columnspan=5, rowspan=5)

labelRaw.grid(row=1, column=14, columnspan=5)
canvasRaw.get_tk_widget().grid(row=2, column=14, columnspan=5, rowspan=5)

labelOutput.grid(row=7, column=8, columnspan=5)
canvasOutput.get_tk_widget().grid(row=9, column=8, columnspan=5, rowspan=5)

labelOverlay.grid(row=7, column=14, columnspan=8)
canvasOverlay.get_tk_widget().grid(row=9, column=14, columnspan=5, rowspan=5)

labelTreeSegments.grid(row=2, column=0, columnspan=6)  #
treeSegments.grid(row=3, column=0, columnspan=6, rowspan=2)  #
labelTreeOutputs.grid(row=10, column=0, columnspan=6)  #
treeOutput.grid(row=11, column=0, columnspan=6, rowspan=2)  #

checkOverwrite.grid(row=5, column=4)

buttonScribble.grid(row=6, column=0, columnspan=2)
textScribbleSize.grid(row=6, column=2)
buttonClear.grid(row=6, column=4, columnspan=3)

labelTextX.grid(row=7, column=0)  #
textX.grid(row=7, column=1)  #
labelTextY.grid(row=7, column=2)  #
textY.grid(row=7, column=3)  #
comboClass.grid(row=7, column=4)
buttonClass.grid(row=7, column=5)
buttonSetAll.grid(row=7, column=6)

buttonDrawBox.grid(row=9, column=0, columnspan=2)
buttonDeleteBox.grid(row=9, column=2, columnspan=2)

labelTextNumClasses.grid(row=9, column=4, sticky="E")
TextNumClasses.grid(row=9, column=5)
buttonCluster.grid(row=9, column=6)

buttonSave.grid(row=15, column=0, columnspan=3)
buttonReset.grid(row=15, column=4, columnspan=3)

textStatus.grid(row=16, column=8, columnspan=17)

frame.grid_rowconfigure(2, weight=0)
frame.grid_rowconfigure(9, weight=0)

frame.grid_rowconfigure(15, minsize=10)

frame.grid_columnconfigure(0, weight=0)

frame.grid_columnconfigure(7, minsize=20)
frame.grid_columnconfigure(13, minsize=20)

frame.grid(padx=20, pady=20)

frame.mainloop()
