import json
import webbrowser
import os
import random
import time
import csv
import numpy as np
from copy import deepcopy
from datetime import datetime
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, f1_score

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from matplotlib.colors import LinearSegmentedColormap

from classes.u_net import UNET
from classes.loss import FocalLoss, IouLoss
from classes.dataloader import ImageDataSet
from classes.splitter import ImageSplitter

import base.connect_to_db as ctd

#from database_connection import Connector

#with open('../../params1.json') as json_file:
#    params1 = json.load(json_file)

# 0=ice, 1=snow, 2=rocks, 3=water, 4=clouds, 5=sky, 6=unknown

print("LOCAL")

continue_training = False

if continue_training:
    continue_model_path = "/home/fdahle/Desktop/ATM/data/models/segmentation/UNET/models_server/training_cropped_min_2_temp.pth"

    print("Continue training")

# model settings
MODEL_NAME = "test"
BINARY = False
aug_type = "resized"  # can be 'resized' or 'cropped'
crop_method = None  # only necessary for crops, possible values are 'weighted', 'inverse', 'random', 'equally'
img_size = 1200 # can be for crops (usually 512) or for resized (usually 1200)
metadata = []  # ["view_direction"]# [] if no metadata
augmentations = ["flipping", "rotation" , "brightness", "noise"] # , "normalize"]

print("MODEL: {}".format(MODEL_NAME))

# debug settings
run_tensor_board = False
bool_debug_mode = True  # that means nothing is saved or written to tensor
max_images = 6  # None if no max images
bool_plot_train = False
bool_plot_valid = False
bool_visual_training = False  # specifies if tensorflow webbrowser is opened
print_batches = False

ids_train_images = []
ids_test_images = []

# advanced model settings
NR_EPOCHS = 10000
LEARNING_RATE = 0.0001
BATCH_SIZE_TRAINING = 4
BATCH_SIZE_VALIDATION = 4
EARLY_STOPPING = 2500  # possible values are integer, if 0 there's no early stopping
LOSS_TYPE = "iou"  # possible values are focal, crossentropy, iou
LOSS_WITH_WEIGHTS = True
KERNEL_SIZE = 3
OUTPUT_LAYERS = 7

evaluation_metrics = ["confusion_matrix", "IoU"]

# splitting settings
split_type = "stratified"  # values can be 'stratified' or 'random'
train_perc = 80
val_perc = 20
test_perc = 0

# some basic settings
verbose = True
seed = 123
fraction = 5
edge = 75  # remove outer part of images as there the segmentation is not good
SAVE_STEPS = 25
min_prob = 0.5  # minimum probability that is needed in order to assign not unknown to a class

# settings to plot and save stuff
plot_overview = False
save_overview = False
plot_graphs = False
save_graphs = False
plot_test = False
save_test = False
create_metrics = False
save_metrics = False
save_stats = True
save_model = True

# train with gpu or cpu
DEVICE = "auto"  # possible values are auto, cpu, gpu

# path to folders OLD
"""
IMG_FOLDER = "../../../../data/aerial/TMA/downloaded"
MASK_FOLDER = "../../../../data/aerial/TMA/segmented/supervised"
base_folder = "../../../../data/models/segmentation/UNET"
SAVE_FOLDER = base_folder + "/models"
STATISTICS_FOLDER = base_folder + "/statistics"
OVERVIEW_FOLDER = base_folder + "/overviews"
GRAPH_FOLDER = base_folder + "/graphs"
METRICS_FOLDER = base_folder + "/metrics"
TEST_FOLDER = base_folder + "/tests"
"""

IMG_FOLDER = "/data_1/ATM/data_1/aerial/TMA/downloaded"
MASK_FOLDER = "/data_1/ATM/data_1/aerial/TMA/masked"
base_folder = "../../../../data/models/segmentation/UNET"
SAVE_FOLDER = ""
STATISTICS_FOLDER = ""
OVERVIEW_FOLDER = ""
GRAPH_FOLDER = ""
METRICS_FOLDER = ""
TEST_FOLDER = ""

# for tensorflow
tracking_address = "../../../../data/models/segmentation/UNET/logs"
run_tracking_address = tracking_address + "/" + MODEL_NAME

# 0 - ice, 1 - snow, 2 - rocks, 3 - water, 4 - clouds, 5 - sky, 6 - unknown


print("DEBUG CUDA")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

"""
# own print function that only prints if verbose is True
def print_verbose(text, **kwarg):
    if verbose is False:
        return

    # distinguish between end or not
    if "end" in kwarg:
        print(text, end=kwarg["end"])
    else:
        print(text)
"""

def get_file_list(path):

    # get the manually labeled images
    sql_string = "SELECT image_id FROM images_segmentation WHERE labelled_by='manual'"
    data = ctd.get_data_from_db(sql_string)

    # Convert the 'image_id' column of the DataFrame to a set for faster lookup
    image_id_set = set(data['image_id'])

    # check if the images are also in the fld
    init_files_list = []
    for filename in os.listdir(path):
        if filename.endswith(".tif"):
            file_id = filename.split(".")[0]

            # Add file ID to the list only if it's in the 'image_id' set
            if file_id in image_id_set:
                init_files_list.append(file_id)

    return init_files_list

def get_borders(files_list):

    files_string = str(files_list)
    files_string = files_string.replace("[", "(")
    files_string = files_string.replace("]", ")")

    # Sort the list
    files_list.sort()

    sql_string = "SELECT image_id, " + \
                 "image_width, image_height, " + \
                 "fid_mark_1_x, " + \
                 "fid_mark_1_y, " + \
                 "fid_mark_2_x, " + \
                 "fid_mark_2_y, " + \
                 "fid_mark_3_x, " + \
                 "fid_mark_3_y, " + \
                 "fid_mark_4_x, " + \
                 "fid_mark_4_y " + \
                 "FROM images_fid_points " + \
                 "WHERE image_id IN " + files_string

    # get data from table
    table_data = ctd.get_data_from_db(sql_string, catch=False)

    if table_data is None:
        print("There is a problem with the table data. Please check your code")
        exit()

    # remove NaN values
    #table_data.dropna(inplace=True)

    borders_dict = {}
    for _, row in table_data.iterrows():

        # get the image id
        image_id = row["image_id"]

        # fill None values with NaN
        row.fillna(value=np.nan, inplace=True)

        # create an empty borders dict
        borders_dict[image_id] = {}

        # get left
        if row["fid_mark_1_x"] >= row["fid_mark_3_x"]:
            left = row["fid_mark_1_x"]
        else:
            left = row["fid_mark_3_x"]

        # get top
        if row["fid_mark_2_y"] >= row["fid_mark_3_y"]:
            top = row["fid_mark_2_y"]
        else:
            top = row["fid_mark_3_y"]

        # get right
        if row["fid_mark_2_x"] <= row["fid_mark_4_x"]:
            right = row["fid_mark_2_x"]
        else:
            right = row["fid_mark_4_x"]

        # get bottom
        if row["fid_mark_1_y"] <= row["fid_mark_4_y"]:
            bottom = row["fid_mark_1_y"]
        else:
            bottom = row["fid_mark_4_y"]

        if np.isnan(left):
            left = 700
        if np.isnan(top):
            top = 700
        if np.isnan(right):
            right = row["image_width"] - 700
        if np.isnan(bottom):
            bottom = row["image_height"] - 700

        try:
            borders_dict[image_id]["left"] = int(left)
            borders_dict[image_id]["right"] = int(right)
            borders_dict[image_id]["top"] = int(top)
            borders_dict[image_id]["bottom"] = int(bottom )
        except:
            continue

    return borders_dict


# function to calculate the evaluation metric
def eval_fn(predb, yb):
    equal = predb == yb

    # in order to make the interpreter clear that this is a np array
    equal = np.array(equal)

    true_vals = equal.sum()
    all_vals = equal.size
    accuracy = true_vals / all_vals

    f1_score_val = f1_score(yb.flatten(), predb.flatten(), average="weighted")

    return accuracy, f1_score_val


# train the actual model
def train(tensor_writer, model, input_train_dl, input_valid_dl,
          optimizer, input_loss_fn, input_eval_fn,
          input_cmap_output, input_norm_output, input_bool_debug_mode,
          starting_epoch, starting_loss, epochs=100):

    print('-' * 10)
    print("Start training on {}:".format(device))

    # track starting time
    start = time.time()

    # set model to device
    model = model.to(device)

    # save the values per epoch
    train_loss_epoch = []
    train_acc_epoch = []
    train_f1_epoch = []
    val_loss_epoch = []
    val_acc_epoch = []
    val_f1_epoch = []

    # to save the best values and their pos
    train_best_loss = 0
    train_best_loss_epoch = 0
    train_best_acc = 0.0
    train_best_acc_epoch = 0
    train_best_f1 = 0.0
    train_best_f1_epoch = 0
    val_best_loss = 100.0
    val_best_loss_epoch = 0
    val_best_acc = 0.0
    val_best_acc_epoch = 0
    val_best_f1 = 0.0
    val_best_f1_epoch = 0

    best_model_dict = None

    epoch = starting_epoch

    now = datetime.now().strftime("%Y%m%d-%H%M%S")

    # training not a for loop, so that ctrl + c can be catched
    continue_training = True
    while continue_training:

        try:
            print('Epoch {}/{}'.format(epoch+1, epochs))

            train_loss_batch = []
            train_acc_batch = []
            train_f1_batch = []

            step = 0
            # iterate over data per batch
            for input_train_x, input_train_y in input_train_dl:

                if bool_plot_train:
                    f, axarr = plt.subplots(2, input_train_x.shape[0])
                    for i in range(input_train_x.shape[0]):
                        axarr[0, i].imshow(input_train_x[i, 0, :, :], cmap="gray")
                        axarr[1, i].imshow(input_train_y[i, :, :], cmap=cmap_output, norm=norm_output,
                                           interpolation='none')
                        axarr[0, i].axis('off')
                        axarr[1, i].axis('off')
                    plt.show()

                # show the first images of the model
                # if step==0:
                #    grid = torchvision.utils.make_grid(input_train_x)
                #    tensor_writer.add_image("images", grid, global_step=epoch)

                step += 1

                # data to gpu
                input_train_x = input_train_x.to(device)
                input_train_y = input_train_y.to(device)

                # zero the gradients
                optimizer.zero_grad()

                # predict
                train_preds = model(input_train_x)

                # get loss
                train_loss = input_loss_fn(train_preds, input_train_y)

                # backpropagation
                train_loss.backward()
                optimizer.step()

                if BINARY:
                    classes = (train_preds > 0.5).cpu().detach().numpy()
                else:
                    classes = np.argmax(train_preds.cpu().detach().numpy(), axis=1)

                # get accuracy
                train_acc, train_f1 = input_eval_fn(classes, input_train_y.cpu().detach().numpy())

                # save the values
                train_loss_batch.append(train_loss.item())
                train_acc_batch.append(train_acc.item())
                train_f1_batch.append(train_f1.item())

                if device == "cpu":
                    if print_batches:
                        print('  Current batch: {} - Loss: {} - Acc: {} - F1: {}'.format(
                            step,
                            round(train_loss.item(), fraction),
                            round(train_acc.item(), fraction),
                            round(train_f1.item(), fraction)
                        ))
                elif device == "cuda":
                    if print_batches:
                        print('  Current batch: {} - Loss: {} - Acc: {} - F1: {} - AllocMem (Mb): {}'.format(
                            step,
                            round(train_loss.item(), fraction),
                            round(train_acc.item(), fraction),
                            round(train_f1.item(), fraction),
                            round(torch.cuda.memory_allocated() / 1024 / 1024, 4)
                        ))

            # get averages of loss and acc for epoch
            train_avg_loss_epoch = sum(train_loss_batch) / len(train_loss_batch)
            train_avg_acc_epoch = sum(train_acc_batch) / len(train_acc_batch)
            train_avg_f1_epoch = sum(train_f1_batch) / len(train_f1_batch)

            # save the best values
            if train_avg_loss_epoch < train_best_loss:
                train_best_loss = train_avg_loss_epoch
                train_best_loss_epoch = epoch
            if train_avg_acc_epoch > train_best_acc:
                train_best_acc = train_avg_acc_epoch
                train_best_acc_epoch = epoch
            if train_avg_f1_epoch < train_best_f1:
                train_best_f1 = train_avg_f1_epoch
                train_best_f1_epoch = epoch

            print(' Current epoch: {} - avg. train Loss: {} - avg. train Acc: {} - avg. train F1: {}'.format(
                epoch+1,
                round(train_avg_loss_epoch, fraction),
                round(train_avg_acc_epoch, fraction),
                round(train_avg_f1_epoch, fraction)
            ))

            # release
            del train_preds
            del train_loss
            del train_acc
            del train_f1
            del classes

            train_loss_epoch.append(train_avg_loss_epoch)
            train_acc_epoch.append(train_avg_acc_epoch)
            train_f1_epoch.append(train_avg_f1_epoch)

            if input_bool_debug_mode is False:
                tensor_writer.add_scalar(MODEL_NAME + "_" + now + '/Loss/train/', train_avg_loss_epoch, epoch+1)
                tensor_writer.add_scalar(MODEL_NAME + "_" + now + '/Accuracy/train/', train_avg_acc_epoch, epoch+1)
                tensor_writer.add_scalar(MODEL_NAME + "_" + now + '/F1_Score/train/', train_avg_f1_epoch, epoch+1)

            val_loss_batch = []
            val_acc_batch = []
            val_f1_batch = []

            step = 0

            with torch.no_grad():  # otherwise the memory blows up
                for valid_x, valid_y in input_valid_dl:

                    if bool_plot_valid:
                        f, axarr = plt.subplots(2, valid_x.shape[0])
                        for i in range(valid_x.shape[0]):
                            axarr[0, i].imshow(valid_x[i, 0, :, :], cmap='gray')
                            axarr[1, i].imshow(valid_y[i, :, :], cmap=input_cmap_output, norm=input_norm_output,
                                               interpolation='none')
                            axarr[0, i].axis('off')
                            axarr[1, i].axis('off')
                        plt.axis('off')
                        plt.show()

                    step += 1

                    # data to gpu
                    valid_x = valid_x.to(device)
                    valid_y = valid_y.to(device)

                    # get values
                    val_preds = model(valid_x)

                    # get loss
                    val_loss = input_loss_fn(val_preds, valid_y)

                    if BINARY:
                        classes = (val_preds > 0.5).cpu().detach().numpy()
                    else:
                        classes = np.argmax(val_preds.cpu().detach().numpy(), axis=1)

                    val_acc, val_f1 = input_eval_fn(classes, valid_y.cpu().detach().numpy())

                    # save the values
                    val_loss_batch.append(val_loss.item())
                    val_acc_batch.append(val_acc.item())
                    val_f1_batch.append(val_f1.item())

                    if device == "cpu":
                        print('  Current batch: {} - Loss: {} - Acc: {} - F1: {}'.format(
                            step,
                            round(val_loss.item(), fraction),
                            round(val_acc.item(), fraction),
                            round(val_f1.item(), fraction)
                        ))
                    elif device == "cuda":
                        print('  Current batch: {} - Loss: {} - Acc: {} - F1: {} -  AllocMem (Mb): {}'.format(
                            step,
                            round(val_loss.item(), fraction),
                            round(val_acc.item(), fraction),
                            round(val_f1.item(), fraction),
                            round(torch.cuda.memory_allocated() / 1024 / 1024, 4)
                        ))

                # get averages of loss and acc for epoch
                val_avg_loss_epoch = sum(val_loss_batch) / len(val_loss_batch)
                val_avg_acc_epoch = sum(val_acc_batch) / len(val_acc_batch)
                val_avg_f1_epoch = sum(val_f1_batch) / len(val_f1_batch)

                if val_avg_loss_epoch < val_best_loss:
                    val_best_loss = val_avg_loss_epoch
                    val_best_loss_epoch = epoch
                    # cpu_dict = {k:v.to('cpu') for k, v in model.state_dict().items()}
                    best_model_dict = deepcopy(model.state_dict())

                if val_avg_acc_epoch > val_best_acc:
                    val_best_acc = val_avg_acc_epoch
                    val_best_acc_epoch = epoch

                if val_avg_f1_epoch < val_best_f1:
                    val_best_f1 = val_avg_f1_epoch
                    val_best_f1_epoch = epoch

                print(' Current epoch: {} - avg. val Loss: {} - avg. val Acc: {} - avg. val F1: {}'.format(
                    epoch+1,
                    round(val_avg_loss_epoch, fraction),
                    round(val_avg_acc_epoch, fraction),
                    round(val_avg_f1_epoch, fraction)
                ))

                # release
                del val_preds
                del val_loss
                del val_acc
                del val_f1
                del classes

                val_loss_epoch.append(val_avg_loss_epoch)
                val_acc_epoch.append(val_avg_acc_epoch)
                val_f1_epoch.append(val_avg_f1_epoch)

                if input_bool_debug_mode is False:
                    tensor_writer.add_scalar(MODEL_NAME + '_' + now + '/Loss/validation/',
                                             val_avg_loss_epoch, epoch+1)
                    tensor_writer.add_scalar(MODEL_NAME + '_' + now + '/Accuracy/validation/',
                                             val_avg_acc_epoch, epoch+1)
                    tensor_writer.add_scalar(MODEL_NAME + '_' + now + '/F1_Score/validation/',
                                             val_avg_f1_epoch, epoch+1)

            if EARLY_STOPPING > 0:
                if epoch - val_best_loss_epoch >= EARLY_STOPPING:
                    print("Early stopping as no improvement in loss in {} rounds. "
                          "Best loss: {}".format(EARLY_STOPPING, str(val_best_loss)))

                    save_path = SAVE_FOLDER + "/" + MODEL_NAME + "_no_early_stopping.pth"

                    if os.path.exists(save_path):
                        os.remove(save_path)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_avg_loss_epoch
                    }, save_path)

                    model.load_state_dict(best_model_dict)
                    continue_training = False

            if SAVE_STEPS > 0:
                if epoch > 0 and epoch % SAVE_STEPS == 0:
                    print("Save intermediate model")

                    save_path = SAVE_FOLDER + "/" + MODEL_NAME + "_temp.pth"

                    if os.path.exists(save_path):
                        os.remove(save_path)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_avg_loss_epoch
                    }, save_path)

            epoch = epoch + 1

            if epoch == epochs:
                continue_training = False

        except KeyboardInterrupt:
            continue_training = False
            input_required = True
            while input_required:
                input_save = input("Do you want to save the model ('y', 'n')")
                if input_save == "y":
                    input_bool_debug_mode = False
                    input_required = False
                elif input_save == "n":
                    input_bool_debug_mode = True
                    input_required = False
                else:
                    print("Please enter 'y' or 'n'")

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    data = {
        "loss_train": train_loss_epoch,
        "acc_train": train_acc_epoch,
        "f1_train": train_f1_epoch,
        "loss_val": val_loss_epoch,
        "acc_val": val_acc_epoch,
        "f1_val": val_f1_epoch,
        "final_epoch": val_best_loss_epoch + 1,
        "train_best_loss_epoch": train_best_loss_epoch,
        "train_best_acc_epoch": train_best_acc_epoch,
        "train_best_f1_epoch": train_best_f1_epoch,
        "val_best_loss_epoch": val_best_loss_epoch,
        "val_best_acc_epoch": val_best_acc_epoch,
        "val_best_f1_epoch": val_best_f1_epoch
    }

    return data, input_bool_debug_mode


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


def create_overview(data, custom_cmap, custom_norm, plot, save, title=""):
    if plot is False and save is False:
        return

    fig, ax = plt.subplots(4, 4)

    i, j = 0, 0

    for elem in data:

        elem_x = elem[0]
        elem_y = elem[1]

        elem_x = elem_x.numpy()[0]
        elem_y = elem_y.numpy()

        ax[i, j].imshow(elem_x, cmap="gray")
        j += 1
        ax[i, j].imshow(elem_y, cmap=custom_cmap, norm=custom_norm, interpolation='none')

        j += 1

        if j == 4:
            i += 1
            j = 0

        if i == 4:
            break

    if title != "":
        plt.suptitle(title)

    if save:
        plt.savefig(OVERVIEW_FOLDER + "/" + title + "_" + MODEL_NAME + ".png")

    if plot:
        plt.show()

    plt.close(fig)


def create_graphs(data_dict, plot, save):
    if plot is False and save is False:
        return

    fig, ax_loss = plt.subplots()

    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")

    ax_acc = ax_loss.twinx()
    ax_acc.set_ylabel("Accuracy")

    ax_loss.xaxis.get_major_locator().set_params(integer=True)
    ax_acc.xaxis.get_major_locator().set_params(integer=True)

    counter = 0
    for key, elem in data_dict.items():

        color = None
        if key.endswith("val"):
            if key.startswith("loss"):
                color = "red"
            if key.startswith("acc"):
                color = "blue"
        if key.endswith("train"):
            if key.startswith("loss"):
                color = "indianred"
            if key.startswith("acc"):
                color = "cornflowerblue"

        if key.startswith("loss"):
            ax_loss.plot(elem, label=key, color=color)
        if key.startswith("acc"):
            ax_acc.plot(elem, label=key, color=color)
        counter += 1

    ax_loss.legend(loc="upper left")
    ax_acc.legend(loc="upper right")

    if EARLY_STOPPING > 0 and len(next(iter(data_dict.values()))) != NR_EPOCHS:
        nr_epochs = len(next(iter(data_dict.values()))) - 1
        early_stopping_pos = nr_epochs - EARLY_STOPPING
        # get nr of epochs
        plt.axvline(x=early_stopping_pos, color='red', ls="--")

    if save:
        plt.savefig(GRAPH_FOLDER + "/" + MODEL_NAME + ".png")

    if plot:
        plt.show()

    plt.close(fig)


def create_tests_and_metrics(model, data, custom_cmap, custom_norm, plot_test, save_test):
    if plot_test is False and save_test is False:
        return

    cmap_lin = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    norm_lin = matplotlib.colors.Normalize(vmin=0, vmax=1)

    i = 0

    cm_total = []
    iou_total = []
    f1_total = []

    with torch.no_grad():
        for data_x, data_y in data:

            # data to device
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            # get values
            preds = model(data_x)

            if BINARY:
                classes = (preds > 0.5).cpu().detach().numpy()[0, 0, :, :]
            else:
                preds = preds.cpu().detach().numpy()

                classes = np.argmax(preds, axis=1)[0, :, :]

                max_probs = np.max(preds, axis=1)[0, :, :]

                classes[max_probs < min_prob] = 6

            # convert data
            data_x = data_x.cpu().detach().numpy()[0, 0, :, :]
            data_y = data_y.cpu().detach().numpy()[0, :, :]

            # create plots

            fig, ax = plt.subplots(2, 2)

            ax[0, 0].imshow(data_x, cmap="gray")
            ax[0, 0].set_title("Image")

            ax[0, 1].imshow(data_y, cmap=custom_cmap, norm=custom_norm, interpolation='none')
            ax[0, 1].set_title("Ground truth")

            ax[1, 0].imshow(max_probs, cmap=cmap_lin, norm=norm_lin, interpolation='none')
            ax[1, 0].set_title("Probability")

            ax[1, 1].imshow(classes, cmap=custom_cmap, norm=custom_norm, interpolation='none')
            ax[1, 1].set_title("Prediction")

            if save_test:
                plt.savefig(TEST_FOLDER + "/" + MODEL_NAME + "_" + str(i) + ".png")

            if plot_test:
                plt.show()

            data_y = data_y.flatten()
            classes = classes.flatten()

            # create metrics
            if "confusion_matrix" in evaluation_metrics:
                my_bins = range(0, 8)
                cm = confusion_matrix(data_y, classes, labels=my_bins)
                cm_total.append(cm)

            if "IoU" in evaluation_metrics:
                my_bins = range(0, 8)
                cm = confusion_matrix(data_y, classes, labels=my_bins)

                false_pos = cm.sum(axis=0) - np.diag(cm)
                false_neg = cm.sum(axis=1) - np.diag(cm)
                true_pos = np.diag(cm)
                # TN = cm.values.sum() - (FP + FN + TP)

                iou = np.array(100 * true_pos / (true_pos + false_neg + false_pos))
                iou_total.append(iou)

            if "f1" in evaluation_metrics:
                f1_score_val = f1_score(data_y, classes)
                f1_total.append(f1_score_val)

            i = i + 1

        if "confusion_matrix" in evaluation_metrics:
            cm_total = np.array(cm_total)
            cm_total = np.sum(cm_total, axis=0)
            print("Confusion matrix total:")
            print(cm_total)

        if "IoU" in evaluation_metrics:
            iou_total = np.array(iou_total)
            iou_avg = np.nanmean(iou_total, axis=0)
            print("IoU:")
            print(iou_avg)

        if "f1" in evaluation_metrics:
            f1_total = np.array(f1_total)
            f1_avg = np.nanmean(f1_total, axis=0)
            print("F1-Score:")
            print(f1_avg)

    plt.close(fig)


def create_stats(nr_images_dict, data_dict):
    if save_stats is False:
        return

    dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M")

    metadata_string = "(" + ",".join(metadata) + ")"
    augmentations_string = "(" + ",".join(augmentations) + ")"

    if metadata_string == "()":
        metadata_string = "None"

    if augmentations_string == "()":
        augmentations_string = "None"

    print("WORKAROUND WITH CROP METHOD")
    crop_method = "None"

    fields = [
        MODEL_NAME,
        dt_string,
        BINARY,
        metadata_string,
        str(nr_images_dict[0]),  # nr of images
        split_type,
        str(train_perc),
        str(nr_images_dict[1]),  # train images
        str(val_perc),
        str(nr_images_dict[2]),  # val images
        str(test_perc),
        str(nr_images_dict[3]),  # test images
        str(edge),
        aug_type,
        str(img_size),
        crop_method,
        augmentations_string,
        str(NR_EPOCHS),
        str(data_dict["final_epoch"]),
        str(LEARNING_RATE),
        str(BATCH_SIZE_TRAINING),
        str(BATCH_SIZE_VALIDATION),
        str(EARLY_STOPPING),
        LOSS_TYPE,
        str(KERNEL_SIZE),
        str(OUTPUT_LAYERS),
        str(round(data_dict["loss_train"][-1], fraction)),
        str(round(data_dict["train_best_loss_epoch"], fraction)),
        str(round(data_dict["acc_train"][-1], fraction)),
        str(round(data_dict["train_best_acc_epoch"], fraction)),
        str(round(data_dict["f1_train"][-1], fraction)),
        str(round(data_dict["train_best_f1_epoch"], fraction)),
        str(round(data_dict["loss_val"][-1], fraction)),
        str(round(data_dict["val_best_loss_epoch"], fraction)),
        str(round(data_dict["acc_val"][-1], fraction)),
        str(round(data_dict["val_best_acc_epoch"], fraction)),
        str(round(data_dict["f1_val"][-1], fraction)),
        str(round(data_dict["val_best_f1_epoch"], fraction))
    ]

    with open(STATISTICS_FOLDER + '/statistics.csv', 'a', newline='') as fd:
        writer = csv.writer(fd, delimiter=";")
        writer.writerow(fields)


if __name__ == "__main__":

    if bool_debug_mode is False:
        # set folder for tensorboard
        if os.path.exists(run_tracking_address) is False:
            os.makedirs(run_tracking_address)

        """
        # init tensorboard
        if run_tensor_board:
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', tracking_address, '--reload_multifile', 'True', '--load_fast', 'false'])
            url = tb.launch()
            print(f"Tensorflow listening on {url}")
            if bool_visual_training:
                webbrowser.open(url, new=1, autoraise=True)
        """

        # writer for pytorch
        t_writer = SummaryWriter(log_dir=run_tracking_address)
    else:
        t_writer = None

    # set seed
    random.seed(seed)
    torch.manual_seed(seed)

    # get custom cmap for colors
    cmap_output, norm_output = create_cmap()

    device = None
    if DEVICE == "cpu":
        device = 'cpu'
    elif DEVICE == "gpu":
        device = 'cuda'
    elif DEVICE == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # save the number of images for later stats
    nr_images_dict = []

    # get the files
    filesList = get_file_list(MASK_FOLDER)

    # get the borders (the part of the images that will be cut off)
    borders = get_borders(filesList)

    # remove images without borders
    filesList = [id for id in filesList if id in borders]

    # save nr of images
    nr_images_dict.append(len(filesList))

    # split images
    percentages = [train_perc, val_perc, test_perc]

    splitter = ImageSplitter(filesList, percentages, split_type, max_images, seed=seed)

    files_list_train, files_list_val, files_list_test = splitter.get_splitted()

    nr_images_dict.append(len(files_list_train))
    nr_images_dict.append(len(files_list_val))
    nr_images_dict.append(len(files_list_test))

    train_ds = ImageDataSet(IMG_FOLDER, MASK_FOLDER, files_list_train, aug_type=aug_type, crop_method=crop_method,
                            img_size=img_size, borders=borders, edge=edge, metadata=metadata)
    valid_ds = ImageDataSet(IMG_FOLDER, MASK_FOLDER, files_list_val, aug_type=aug_type,
                            img_size=img_size, borders=borders, edge=edge, bool_train=False, metadata=metadata)
    test_ds = ImageDataSet(IMG_FOLDER, MASK_FOLDER, files_list_test, aug_type=aug_type,
                           img_size=img_size, borders=borders, edge=edge, bool_train=False, metadata=metadata)

    if LOSS_WITH_WEIGHTS:
        weights = train_ds.get_weights(OUTPUT_LAYERS)
        weights = torch.FloatTensor(weights)
        weights = weights.to(device)
    else:
        weights = None

    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    train_params = {
        "batch_size": BATCH_SIZE_TRAINING,
        "shuffle": True,
        "num_workers": 2,
        "worker_init_fn": seed_worker,
        "generator": g
    }

    val_params = {
        "batch_size": BATCH_SIZE_VALIDATION,
        "shuffle": True,
        "num_workers": 2,
        "worker_init_fn": seed_worker,
        "generator": g
    }

    test_params = {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 2,
        "worker_init_fn": seed_worker,
        "generator": g
    }

    # create dataloaders
    train_dl = DataLoader(train_ds, **train_params)
    valid_dl = DataLoader(valid_ds, **val_params)
    if test_perc > 0:
        test_dl = DataLoader(test_ds, **test_params)

    input_layers = 1 + len(metadata)

    # specify the model
    unet = UNET(input_layers, OUTPUT_LAYERS, kernel_size=KERNEL_SIZE, binary=BINARY)

    # specify the loss
    loss_fn = None
    if LOSS_TYPE == "crossentropy":
        if BINARY:
            loss_fn = nn.BCELoss()
        else:
            loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=6)
    elif LOSS_TYPE == "focal":
        loss_fn = FocalLoss(alpha=weights, gamma=2, ignore_index=6)
    elif LOSS_TYPE == "iou":
        loss_fn = IouLoss(ignore_index=6)

    # specify the optimizer
    optim = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)

    starting_epoch = 0
    starting_loss = 0

    # continue training
    if continue_training:
        checkpoint = torch.load(continue_model_path, map_location=device)

        unet.to(device)

        unet.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        starting_epoch = checkpoint['epoch']
        starting_loss = checkpoint['loss']


    # train the model
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="UserWarning: Named tensors and all their associated APIs")
        dataDict, bool_debug_mode = train(t_writer, unet, train_dl, valid_dl, optim, loss_fn, eval_fn, cmap_output,
                                          norm_output, bool_debug_mode, starting_epoch, starting_loss, epochs=NR_EPOCHS)

    if bool_debug_mode is False and t_writer is not None:
        # close writer
        t_writer.close()

    # save model
    if save_model and bool_debug_mode is False:

        temp_save_path = SAVE_FOLDER + "/" + MODEL_NAME + "_temp.pth"

        if os.path.exists(temp_save_path):
            os.remove(temp_save_path)

        torch.save(unet, SAVE_FOLDER + "/" + MODEL_NAME + ".pt")

    # evaluate the model
    if bool_debug_mode is False:
        create_graphs(dataDict, plot_graphs, save_graphs)

    # visualize the tests
    if bool_debug_mode is False and test_perc > 0:
        create_tests_and_metrics(unet, test_dl, cmap_output, norm_output, plot_test, save_test)

    # save the stats
    if bool_debug_mode is False:
        create_stats(nr_images_dict, dataDict)
