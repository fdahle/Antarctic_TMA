"""
This algorithm does the following things:
depending on the operation mode it is either
- training the svm model for extracting subsets
- applying the trained model to extract subsets
"""
import sys #in order to import other python files
import cv2 #in order to load the image
import dlib #in order to train and extract
import matplotlib.pyplot as plt

#add for import
sys.path.append('../../misc')

from database_connections import *
from image_loader import *

mode="extraction" #possible values are "training" or "extraction"
overwrite = False

#model params
fid_type=1
fid_directions = ["N", "E", "S", "W"]
crop_size = 0.4
subset_width, subset_height = 250, 250

#debug params
verbose = True
debug_show_hog = True
debug_show_subsets = False
debug_show_crops = False

#path
model_folder_path = "../../../data/models/subset_extraction"

def train_model():

    #get the image parameters from the database
    def get_table_data(fid_direction):

        conn = Connector()

        #specify params for table select
        tables = ["image_properties", "images"]
        fields = [
                  ["photo_id",
                  "subset_width", "subset_height",
                  "subset_"+fid_direction+"_x",
                  "subset_"+fid_direction+"_y"],
                  ["file_path"]
                 ]
        join = ["photo_id"]
        filters = [
            {
            "subset_"+fid_direction+"_x": "NOT NULL",
            "subset_"+fid_direction+"_y": "NOT NULL",
            "fid_type": fid_type
            }
        ]

        #get data from table
        tableData = conn.get_data(tables, fields, filters, join)

        return tableData

    #load the images
    def load_images(imgDict, tableData):

        #iterate all rows
        for idx, row in tableData.iterrows():

            imgId = row['photo_id']
            filePath = row['file_path']

            #only load image if not already in dict
            if imgId not in imgDict:

                img = load_image(row['file_path'])

                if img is None:
                    imgDict[imgId] = None
                    continue

                imgDict[imgId] = img

        return imgDict

    def create_crops(fid_direction, tableData, imgDict):

        crops = []
        cropParams = {}
        for idx, row in tableData.iterrows():

            params = {}

            image = imgDict[row['photo_id']]

            if image is None:
                cropParams[row['photo_id']] = None
                continue

            height, width = image.shape
            params['height'] = height
            params['width'] = width

            mid_y = int(height/2)
            mid_x = int(width/2)
            params["mid_x"] = mid_x
            params["mid_y"] = mid_y

            cropSize_y = int(mid_y * crop_size)
            cropSize_x = int(mid_y * crop_size)
            params["cropSize_y"] = cropSize_y
            params["cropSize_x"] = cropSize_x

            if fid_direction == "N":
                crop_top = 0
                crop_left = int(mid_x - cropSize_x * 0.5)
                crop_bottom = cropSize_y
                crop_right = int(mid_x + cropSize_x * 0.5)
            elif fid_direction == "E":
                crop_top = int(mid_y - cropSize_y * 0.5)
                crop_left = width - cropSize_x
                crop_bottom = int(mid_y + cropSize_y * 0.5)
                crop_right = width
            elif fid_direction == "S":
                crop_top = height - cropSize_y
                crop_left = int(mid_x - cropSize_x * 0.5)
                crop_bottom = height
                crop_right = int(mid_x + cropSize_x * 0.5)
            elif fid_direction == "W":
                crop_top = int(mid_y - cropSize_y * 0.5)
                crop_left = 0
                crop_bottom = int(mid_y + cropSize_y * 0.5)
                crop_right = cropSize_x

            crop = image[crop_top:crop_bottom,crop_left:crop_right]
            crops.append(crop)

            cropParams[row['photo_id']] = params

        return crops, cropParams

    #create the bounding boxes
    def create_boxes(fid_direction, tableData, cropParams):

        boxes = []
        #iterate all rows
        for idx, row in tableData.iterrows():

            params = cropParams[row['photo_id']]

            if params is None:
                continue

            width = params['width']
            height = params['height']

            mid_x = params['mid_x']
            mid_y = params['mid_y']

            cropSize_x = params["cropSize_x"]
            cropSize_y = params["cropSize_y"]

            subset_width = row["subset_width"]
            subset_height = row["subset_height"]

            x1 = row['subset_'+fid_direction+'_x']
            y1 = row['subset_'+fid_direction+'_y']

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0

            #change coordinates due to subsets
            if fid_direction == "N":
                x1 = x1 - (int(mid_x - cropSize_x * 0.5))
            elif fid_direction == "E":
                x1 = x1 - (width - cropSize_x)
                y1 = y1 + cropSize_y - int(mid_y + cropSize_y * 0.5)
            elif fid_direction == "S":
                x1 = x1 - int(mid_x - cropSize_x * 0.5)
                y1 = y1 - cropSize_y - int(mid_y + cropSize_y * 0.5)
            elif fid_direction == "W":
                y1 =  y1 - int(mid_y - cropSize_y * 0.5)


            x2 = x1 + subset_width
            y2 = y1 + subset_height

            dlib_box = [dlib.rectangle(left=x1 , top=y1, right=x2, bottom=y2)]
            boxes.append(dlib_box)
        return boxes

    #save the loaded images (so that they do not need to be loaded again during
    #runtime)
    imgDict = {}

    #iterate all directions, for every sky direction there's an own model
    for fid_direction in fid_directions:

        if verbose:
            print("train detector for " + fid_direction + "..", end='\r')

        #get the information from the sql database
        tableData = get_table_data(fid_direction)

        #load the images and store them in a dict
        imgDict = load_images(imgDict, tableData)

        #select necessary images and crop these (for faster execution time)
        crops, cropParams = create_crops(fid_direction, tableData, imgDict)

        boxes = create_boxes(fid_direction, tableData, cropParams)

        # Initialize object detector Options
        options = dlib.simple_object_detector_training_options()
        options.add_left_right_image_flips = False
        options.C = 5

        #train
        detector = dlib.train_simple_object_detector(crops, boxes, options)
        file_name = 'detector_subset_' + fid_direction + '.svm'
        detector.save(model_folder_path + "/" + file_name)

        if debug_show_hog:
            hog_detector = dlib.image_window()
            hog_detector.set_image(detector)

        if verbose:
            print("train detector for " + fid_direction + ".. - finished sucessfully")

def create_subsets():

    conn = Connector()

    def load_models(folder):

        if verbose:
            print("loading models..", end='\r')

        models = {"N":None,"E":None,"S":None,"W":None}

        for key in models:
            filePath = folder + "/detector_subset_" + key + ".svm"
            model = dlib.simple_object_detector(filePath)
            models[key] = model

        if verbose:
            print("loading models.. - finished")

        return models

    def get_table_data(conn):

        if verbose:
            print("reading ids from table..", end='\r')

        #specify params for table select
        tables = ["image_properties", "images"]
        fields = [["photo_id",
                   "subset_n_x", "subset_n_y",
                   "subset_e_x", "subset_e_y",
                   "subset_s_x", "subset_s_y",
                   "subset_w_x", "subset_w_y",
                  ],
                  ["file_path"]
                  ]
        join = ["photo_id"]
        filters = [
            {"image_properties.subset_n_x":"NULL",
             "image_properties.subset_n_y":"NULL"},
            {"image_properties.subset_e_x":"NULL",
             "image_properties.subset_e_y":"NULL"},
            {"image_properties.subset_s_x":"NULL",
             "image_properties.subset_s_y":"NULL"},
            {"image_properties.subset_w_x":"NULL",
             "image_properties.subset_w_y":"NULL"}
        ]

        if overwrite:
            filters = []

        #get data from table
        tableData = conn.get_data(tables, fields, filters, join)

        if verbose:
            print("reading ids from table.. - finished")

        return tableData

    def get_crops(image, imgId):
        params = {}
        crops = {"N":None,"E":None,"S":None,"W":None}

        height, width = image.shape
        params['height'] = height
        params['width'] = width

        mid_y = int(height/2)
        mid_x = int(width/2)
        params["mid_x"] = mid_x
        params["mid_y"] = mid_y

        cropSize_y = int(mid_y * crop_size)
        cropSize_x = int(mid_y * crop_size)
        params["cropSize_y"] = cropSize_y
        params["cropSize_x"] = cropSize_x

        for key in crops:
            if key == "N":
                crop_top = 0
                crop_left = int(mid_x - cropSize_x * 0.5)
                crop_bottom = cropSize_y
                crop_right = int(mid_x + cropSize_x * 0.5)
            elif key == "E":
                crop_top = int(mid_y - cropSize_y * 0.5)
                crop_left = width - cropSize_x
                crop_bottom = int(mid_y + cropSize_y * 0.5)
                crop_right = width
            elif key == "S":
                crop_top = height - cropSize_y
                crop_left = int(mid_x - cropSize_x * 0.5)
                crop_bottom = height
                crop_right = int(mid_x + cropSize_x * 0.5)
            elif key == "W":
                crop_top = int(mid_y - cropSize_y * 0.5)
                crop_left = 0
                crop_bottom = int(mid_y + cropSize_y * 0.5)
                crop_right = cropSize_x

            crop = image[crop_top:crop_bottom,crop_left:crop_right]
            crops[key] = crop

        if debug_show_crops:
            show_crops(imgId, crops)


        return crops, params

    def show_crops(imgId, crops):
        f, axarr = plt.subplots(2,2)
        f.suptitle(imgId)
        f.tight_layout()
        if crops["N"] is not None:
            axarr[0,0].imshow(crops["N"], cmap='gray')
            axarr[0,0].set_title("N")
        if crops["E"] is not None:
            axarr[0,1].imshow(crops["E"], cmap='gray')
            axarr[0,1].set_title("E")
        if crops["S"] is not None:
            axarr[1,0].imshow(crops["S"], cmap='gray')
            axarr[1,0].set_title("S")
        if crops["W"] is not None:
            axarr[1,1].imshow(crops["W"], cmap='gray')
            axarr[1,1].set_title("W")
        plt.show()

    def find_subsets(models, crops, imgId):

        subsets = {"N":None,"E":None,"S":None,"W":None}

        subsetsFound = 0

        for key in subsets:
            detection = models[key](crops[key])

            #nothing found
            if len(detection) == 0:
                pass
            elif len(detection) == 1:

                subsetsFound = subsetsFound + 1

                x1 = detection[0].left()
                y1 = detection[0].top()
                x2 = detection[0].right()
                y2 = detection[0].bottom()

                subsets[key] = {"x1": x1, "x2": x2, "y1": y1, "y2":y2}

            else:
                #not necessary but to state out that if more than one subset is found
                #that the classifier made an error
                subsets[key] == None

        if debug_show_subsets:
            show_subsets(imgId, subsets)

        return subsets, subsetsFound

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

    def recalculate_subsets(subsets, params):
        for key in subsets:
            subset = subsets[key]

            if subset is None:
                continue

            height = params["height"]
            width = params["width"]

            mid_x = params["mid_x"]
            mid_y = params["mid_y"]

            cropSize_x = params["cropSize_x"]
            cropSize_y = params["cropSize_y"]


            if key == "N":
                subset["x1"] = subset["x1"] + int(mid_x - cropSize_x * 0.5)
                subset["x2"] = subset["x2"] + int(mid_x - cropSize_x * 0.5)
            elif key == "E":
                subset["x1"] = subset["x1"] + width - cropSize_x
                subset["y1"] = subset["y1"] + int(mid_y - cropSize_y * 0.5)
                subset["x2"] = subset["x2"] + width - cropSize_x
                subset["y2"] = subset["y2"] + int(mid_y - cropSize_y * 0.5)
            elif key == "S":
                subset["x1"] = subset["x1"] + int(mid_x - cropSize_x * 0.5)
                subset["y1"] = subset["y1"] + height - cropSize_y
                subset["x2"] = subset["x2"] + int(mid_x - cropSize_x * 0.5)
                subset["y2"] = subset["y2"] + height - cropSize_y
            elif key == "W":
                subset["y1"] = subset["y1"] + int(mid_y - cropSize_y * 0.5)
                subset["y2"] = subset["y2"] + int(mid_y - cropSize_y * 0.5)

            subsets[key] = subset

        return subsets

    def update_table(conn, photo_id, subsets):

        table="image_properties"
        #which entry should be updated
        id = {
            "photo_id": photo_id
        }
        data = {
            "fid_type": 1,
            "subset_width": subset_width,
            "subset_height": subset_height,
            "subset_extraction_date": "DATE()"
         }

        if subsets["N"] is not None:
            data["subset_n_x"] = subsets["N"]["x1"]
            data["subset_n_y"] = subsets["N"]["y1"]
        if subsets["E"] is not None:
            data["subset_e_x"] = subsets["E"]["x1"]
            data["subset_e_y"] = subsets["E"]["y1"]
        if subsets["S"] is not None:
            data["subset_s_x"] = subsets["S"]["x1"]
            data["subset_s_y"] = subsets["S"]["y1"]
        if subsets["W"] is not None:
            data["subset_w_x"] = subsets["W"]["x1"]
            data["subset_w_y"] = subsets["W"]["y1"]


        conn.edit_data(table, id, data)

    models = load_models(model_folder_path)

    #get the ids of the images where subsets are missing
    tableData = get_table_data(conn)

    for idx, row in tableData.iterrows():

        if verbose:
            print("extracing subsets for " + row["photo_id"] + "..", end='\r')

        #load the images
        img = load_image(row['file_path'])

        if img is None:
            print("extracing subsets for " + row["photo_id"] + ".. - Image not found")
            continue

        #get the crops of the images
        crops, params = get_crops(img, row["photo_id"])

        #find subsets
        subsets, subsetsFound = find_subsets(models, crops, row["photo_id"])

        #recalculate to global values
        subsets = recalculate_subsets(subsets, params)

        update_table(conn, row["photo_id"], subsets)

        if verbose:
            print("extracing subsets for " + row["photo_id"] + ".. - finished (" + \
                  str(subsetsFound) + " subsets found)")



if __name__ == "__main__":
    if mode == "training":
        train_model()
    elif mode == "extraction":
        create_subsets()
