import copy
import json
import os
import cv2
import csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.init
from database_connection import Connector
from image_loader import load_image

with open('../../params1.json') as json_file:
    params1 = json.load(json_file)

verbose = True
overwrite = False
debug_visualize = False

path_output_folder = "/home/fdahle/Desktop/ATM/data/aerial/TMA/segmented/unsupervised"


imageSize = params1["unsupervised_image_size"]

minLabels = params1["unsupervised_min_nr_labels"]
maxIter = params1["unsupervised_max_epochs"]
nChannels = params1["unsupervised_nr_channels"]
nConv = params1["unsupervised_n_conv"]
lr = params1["unsupervised_lr"]
earlyStopping = params1["unsupervised_early_stopping"]
stepSizeSim = params1["unsupervised_step_size_sim"]
stepSizeCon = params1["unsupervised_step_size_con"]
stepSizeScr = params1["unsupervised_step_size_scr"]

use_cuda = torch.cuda.is_available()

workType="project"
imageList_path = "/home/fdahle/Desktop/ATM/data/_need_to_check/segment.csv"


# CNN model
class MyNet(nn.Module):
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv - 1):
            self.conv2.append(nn.Conv2d(nChannels, nChannels, kernel_size=(3, 3), stride=(1, 1), padding=1))
            self.bn2.append(nn.BatchNorm2d(nChannels))
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(nChannels)

    def forward(self, x):
        x = self.conv1(x)
        x = func.relu(x)
        x = self.bn1(x)
        for i in range(nConv - 1):
            x = self.conv2[i](x)
            x = func.relu(x)
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    if pad_width[1] != 0:                      # <-- the only change (0 indicates no padding)
        vector[-pad_width[1]:] = pad_value


def create_segmented(workType, imageList_path):
    conn = Connector()

    def get_table_data():

        sql_string = "SELECT images_properties.image_id, " + \
                     "images_properties.fid_mark_1_x, " + \
                     "images_properties.fid_mark_1_y, " + \
                     "images_properties.fid_mark_2_x, " + \
                     "images_properties.fid_mark_2_y, " + \
                     "images_properties.fid_mark_3_x, " + \
                     "images_properties.fid_mark_3_y, " + \
                     "images_properties.fid_mark_4_x, " + \
                     "images_properties.fid_mark_4_y, " + \
                     "images.file_path " + \
                     "FROM images_properties INNER JOIN images " + \
                     "ON (images_properties.image_id = images.image_id AND " + \
                     "images_properties.fid_mark_1_x is NOT NULL AND " + \
                     "images_properties.fid_mark_1_y is NOT NULL AND " + \
                     "images_properties.fid_mark_2_x is NOT NULL AND " + \
                     "images_properties.fid_mark_2_y is NOT NULL AND " + \
                     "images_properties.fid_mark_3_x is NOT NULL AND " + \
                     "images_properties.fid_mark_3_y is NOT NULL AND " + \
                     "images_properties.fid_mark_4_x is NOT NULL AND " + \
                     "images_properties.fid_mark_4_y is NOT NULL) "

        # get data from table
        data = conn.get_data(sql_string)

        return data

    def cut_off_edge(_input_img, input_row):

        input_img = copy.deepcopy(_input_img)

        # get left
        if input_row["fid_mark_1_x"] >= input_row["fid_mark_3_x"]:
            left = input_row["fid_mark_1_x"]
        else:
            left = input_row["fid_mark_3_x"]

        # get top
        if input_row["fid_mark_2_y"] >= input_row["fid_mark_3_y"]:
            top = input_row["fid_mark_2_y"]
        else:
            top = input_row["fid_mark_3_y"]

        # get right
        if input_row["fid_mark_2_x"] <= input_row["fid_mark_4_x"]:
            right = input_row["fid_mark_2_x"]
        else:
            right = input_row["fid_mark_4_x"]

        # get bottom
        if input_row["fid_mark_1_y"] <= input_row["fid_mark_4_y"]:
            bottom = input_row["fid_mark_1_y"]
        else:
            bottom = input_row["fid_mark_4_y"]

        left = int(left)
        right = int(right)
        top = int(top)
        bottom = int(bottom)

        input_img = input_img[top:bottom, left:right]

        return input_img, [top, bottom, left, right]

    def segment(_input_img, filename):

        input_img = copy.deepcopy(_input_img)

        old_dim = (input_img.shape[1], input_img.shape[0])

        # save the dimension and create new dimension
        dim = (imageSize, imageSize)

        # resize
        im = cv2.resize(input_img, dim)

        # stack
        im = np.stack([im, im, im], axis=2)

        # convert data to torch
        data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32') / 255.]))
        if use_cuda:
            data = data.cuda()
        data = Variable(data)

        # train
        model = MyNet(data.size(1))
        if use_cuda:
            model.cuda()
        model.train()

        # similarity loss definition
        loss_fn = torch.nn.CrossEntropyLoss()

        # continuity loss definition
        loss_hpy = torch.nn.L1Loss(size_average=True)
        loss_hpz = torch.nn.L1Loss(size_average=True)

        h_py_target = torch.zeros(im.shape[0] - 1, im.shape[1], nChannels)
        h_pz_target = torch.zeros(im.shape[0], im.shape[1] - 1, nChannels)
        if use_cuda:
            h_py_target = h_py_target.cuda()
            h_pz_target = h_pz_target.cuda()

        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
        label_colours = np.random.randint(255, size=(100, 3))

        best_labels = 1000
        counter = 0

        for batch_idx in range(maxIter):
            # forwarding
            optimizer.zero_grad()
            output = model(data)[0]
            output = output.permute(1, 2, 0).contiguous().view(-1, nChannels)

            output_hp = output.reshape((im.shape[0], im.shape[1], nChannels))
            h_py = output_hp[1:, :, :] - output_hp[0:-1, :, :]
            h_pz = output_hp[:, 1:, :] - output_hp[:, 0:-1, :]
            lhpy = loss_hpy(h_py, h_py_target)
            lhpz = loss_hpz(h_pz, h_pz_target)

            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            n_labels = len(np.unique(im_target))

            if debug_visualize:
                im_target_rgb = np.array([label_colours[c % nChannels] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
                cv2.imshow("output", im_target_rgb)
                cv2.waitKey(10)

            # loss
            loss = stepSizeSim * loss_fn(output, target) + stepSizeCon * (lhpy + lhpz)

            loss.backward()
            optimizer.step()

            if verbose:
                print(batch_idx, '/', maxIter, '|', ' label num :', n_labels, ' | loss :', loss.item())

            if n_labels <= minLabels:
                if verbose:
                    print("nLabels", n_labels, "reached minLabels", minLabels, ".")
                break

            if n_labels < best_labels:
                best_labels = n_labels
                counter = 0
            else:
                counter += 1

            if counter == earlyStopping:
                if verbose:
                    print("nLabels", n_labels, "reached earlyStopping", earlyStopping, ".")
                break

        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, nChannels)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        im_target = im_target.reshape((im.shape[0], im.shape[1])).astype(np.uint8)

        unique_vals = np.unique(im_target)

        i = 100
        for elem in unique_vals:
            im_target[im_target == elem] = i
            i = i + 1

        im_target = cv2.resize(im_target, old_dim, interpolation=cv2.INTER_NEAREST)

        return im_target

    if verbose:
        print("get data from database..", end='\r')

    table_data = get_table_data()

    if table_data is None:
        if verbose:
            print("reading data from table.. - could not get the data")
        exit()
    else:
        if verbose:
            print("get data from database.. - finished")

    if random:
        table_data = table_data.sample(frac=1).reset_index(drop=True)

    if workType == "project":
        image_list = []
        with open(imageList_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                imageId = row[0].split(".")[0]
                image_list.append(imageId)

        table_data = table_data[table_data["image_id"].isin(image_list)]

    for idx, row in table_data.iterrows():

        already_existing = False
        if os.path.isfile(path_output_folder + "/" + row["image_id"] + ".tif"):
            already_existing = True

        if os.path.isfile(path_output_folder + "/done/" + row["image_id"] + ".tif"):
            already_existing = True

        if os.path.isfile(path_output_folder + "/failed/" + row["image_id"] + ".tif"):
            os.remove(path_output_folder + "/failed/" + row["image_id"] + ".tif")
            already_existing = False

        if overwrite is False and already_existing:
            if verbose:
                print(row["image_id"] + " already segmented")
            continue

        if verbose:
            print("segment " + row["image_id"] + "..")

        # load the images
        img_raw = load_image(row['file_path'])

        if img_raw is None:
            print("something went wrong with {}".format(row["image_id"]))
            continue

        img, dimensions = cut_off_edge(img_raw, row)

        img = img[50:img.shape[0]-50, 50:img.shape[1]-50]

        img_segmented = segment(img, row["image_id"])

        top_add = dimensions[0]
        left_add = dimensions[2]
        bottom_add = img_raw.shape[0] - dimensions[1]
        right_add = img_raw.shape[1] - dimensions[3]

        img_segmented = np.pad(img_segmented, ((50, 50), (50, 50)), pad_with,
                                                padder=7)
        img_segmented = np.pad(img_segmented, ((top_add, bottom_add), (left_add, right_add)), pad_with,
                                                padder=7)

        if img_segmented.shape != img_raw.shape:
            print("SOMETHING WENT WRONG")
            exit()

        cv2.imwrite(path_output_folder + "/" + row["image_id"] + ".tif", img_segmented)


if __name__ == "__main__":
    create_segmented(workType, imageList_path)
