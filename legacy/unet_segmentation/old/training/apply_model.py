import json
import torch
import os
import pathlib
from PIL import Image
import cv2
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import LinearSegmentedColormap
from torch import nn

# 0=ice, 1=snow, 2=rocks, 3=water, 4=clouds, 5=sky, 6=unknown


# noinspection PyUnresolvedReferences
#from train_model import UNET

from database_connection import Connector
from classes.u_net import UNET

with open('../../params1.json') as json_file:
    params1 = json.load(json_file)


MODEL_NAME = "training_resized_min_3_no_early_stopping.pth"
model_location = "server" # can be local or server
IMG_SIZE = (1200, 1200)
edge = 0

metadata = None #["view_direction"]# None if no metadata
if metadata is None:
    num_layers = 1
else:
    num_layers = 1 + len(metadata)
min_prob = 0.5

print(MODEL_NAME)

IMG_FOLDER = "../../../../data/aerial/TMA/downloaded"
MODEL_FOLDER = "../../../../data/models/segmentation/UNET/models"
path_csv = "../../../../data/_need_to_check/segment.csv"

if model_location == "server":
    MODEL_FOLDER = MODEL_FOLDER + "_server"

select_type = "random"  # can be random or id

random_max = 10  # max random images that should be shown
photo_id = ""

conn = Connector()


def create_cmap():
    dark_gray = (150, 149, 158)  # ice
    light_gray = (230, 230, 235)  # snow
    black = (46, 45, 46)  # rocks
    dark_blue = (7, 29, 232)  # water
    light_blue = (25, 227, 224)  # clouds
    dark_red = (186, 39, 32)  # sky
    pink = (224, 7, 224)  # unknown

    # set colors for the plots
    _colors = [dark_gray, light_gray, black, dark_blue, light_blue, dark_red, pink]

    # divide colors by 255 (important for matplotlib
    colors = []
    for elem in _colors:
        col = tuple(ti / 255 for ti in elem)
        colors.append(col)

    limits = range(0, 8)
    custom_cmap, custom_norm = from_levels_and_colors(limits, colors)

    return custom_cmap, custom_norm


def remove(input_img, input_filename):

    short_id = input_filename[:-4]

    sql_string = "SELECT image_id, " + \
                 "fid_mark_1_x, " + \
                 "fid_mark_1_y, " + \
                 "fid_mark_2_x, " + \
                 "fid_mark_2_y, " + \
                 "fid_mark_3_x, " + \
                 "fid_mark_3_y, " + \
                 "fid_mark_4_x, " + \
                 "fid_mark_4_y " + \
                 "FROM images_properties " + \
                 "WHERE image_id='" + short_id + "'"

    # get data from table
    table_data = conn.get_data(sql_string)

    subset_border = params1["unsupervised_subset_border"]

    if table_data is None:
        print("There is a problem with the table data. Please check your code")
        exit()

    try:
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
    except (Exception,):

        # that means one point is missing

        return None

    left = int(left + subset_border)
    right = int(right - subset_border)
    top = int(top + subset_border)
    bottom = int(bottom - subset_border)

    input_img = input_img[top:bottom, left:right]

    return input_img

def add_metadata(img, img_id):

    if metadata is None:
        return img

    img_id = img_id.split(".")[0]

    if "height" in metadata:
        sql_string = "SELECT altitude from Images where image_id='" + img_id + "'"
        data = conn.get_data(sql_string)
        data = data.iloc[0]["altitude"]

    if "view_direction" in metadata:
        sql_string = "SELECT view_direction from Images where image_id='" + img_id + "'"
        data = conn.get_data(sql_string)
        data = data.iloc[0]["view_direction"]
        if data == "V":
            val=0
        else:
            val= 1
        view_direction_channel = torch.from_numpy(np.full(img.shape, val))

        img = torch.cat((img, view_direction_channel), axis=1)

    return img

def get_pred(img, img_id, device):
    if IMG_SIZE is not None:
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
    img = img[edge:img.shape[0] - edge, edge:img.shape[1] - edge]

    img_t = np.expand_dims(img, axis=0)
    img_t = np.expand_dims(img_t, axis=0)

    img=img/128 - 1

    img_t = torch.from_numpy(img_t)
    img_t = img_t.float()

    img_t = add_metadata(img_t, img_id)

    img_t = img_t.to(device)

    with torch.no_grad():

        if os.path.isfile(os.path.realpath(MODEL_FOLDER + "/" + MODEL_NAME)):

            if MODEL_NAME.split(".")[-1] == "pt":
                model = torch.load(MODEL_FOLDER + "/" + MODEL_NAME)
                model.eval()
            elif MODEL_NAME.split(".")[-1] == "pth":
                model = UNET(num_layers, 7)
                model.to(device)
                checkpoint = torch.load(MODEL_FOLDER + "/" + MODEL_NAME)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()


        elif os.path.isfile(os.path.realpath(MODEL_FOLDER + "/" + MODEL_NAME + ".pt")):
            model = torch.load(MODEL_FOLDER + "/" + MODEL_NAME + ".pt")
            model.eval()
        elif os.path.isfile(os.path.realpath(MODEL_FOLDER + "/" + MODEL_NAME + "_temp.pth")):
            model = UNET(num_layers,7)
            model.to(device)
            checkpoint = torch.load(MODEL_FOLDER + "/" + MODEL_NAME + "_temp.pth")
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        else:
            print("No model found")
            exit()

        sigm = nn.Sigmoid()

        pred = model(img_t)
        classes = torch.argmax(pred, dim=1)
        probabilities = torch.amax(pred, dim=1)[0, :, :]
        probabilities = sigm(probabilities)

        classes = classes.cpu().detach().numpy()
        classes = np.squeeze(classes, axis=0)

        probabilities = probabilities.cpu().detach().numpy()



        classes[probabilities < min_prob] = 6

    return classes, probabilities


def update_csv(elem):

    with open(path_csv, 'a') as fd:
            fd.write(elem + "\n")

def press(event):

    event_fig = event.canvas.figure
    event_id = event_fig._suptitle.get_text()
    if event.key == 'e':  # exit
        exit()
    elif event.key == 'y':
        update_csv(event_id)
        plt.close()
    elif event.key == 'n':
        plt.close()
    else:
        pass


def apply_model():
    files_list = []
    if select_type == "random":
        for filename in os.listdir(IMG_FOLDER):
            if filename.endswith(".tif"):
                files_list.append(filename)
    elif select_type == "id":
        files_list.append(photo_id)

    random.shuffle(files_list)

    cmap_output, norm_output = create_cmap()
    cmap_lin = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    norm_lin = matplotlib.colors.Normalize(vmin=0, vmax=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    counter = 0

    for elem in files_list:
        im_path = IMG_FOLDER + "/" + elem
        im = Image.open(im_path)
        img = np.array(im)

        img = remove(img, elem)

        if img is None:
            continue

        classes, probs = get_pred(img, elem, device)

        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(img, cmap='gray')
        axarr[1].imshow(classes, cmap=cmap_output, norm=norm_output, interpolation='none')
        axarr[2].imshow(probs, cmap=cmap_lin) # , norm=norm_lin)


        f.canvas.mpl_connect('key_press_event', press)
        f.suptitle(elem)

        plt.show()

        counter += 1

        if counter == random_max:
            break


if __name__ == "__main__":
    apply_model()
