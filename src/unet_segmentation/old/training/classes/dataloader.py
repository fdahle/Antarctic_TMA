import cv2
import numpy as np
import warnings
import copy
import random

import torch
from torch.utils.data import Dataset

import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image


class ImageDataSet(Dataset):

    def __init__(self, image_folder, label_folder, image_ids,
                 aug_type='resized', crop_method=None, augmentations=[], img_size=512, borders=None, edge=None,
                 metadata = None, bool_binary=False, bool_train=True, seed=123):

        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_ids = image_ids

        self.aug_type = aug_type  # can be 'resized' or 'cropped'
        self.crop_method = crop_method
        self.augmentations = augmentations

        #check crop method
        if self.aug_type == "cropped" and bool_train==True:
            print(self.crop_method)
            if self.crop_method not in ["weighted", "inverse", "random", "equally"]:
                print("A wrong crop method was selected")
                exit()

        self.img_size = img_size

        self.borders = borders
        self.edge = edge

        self.metadata = {}
        if metadata is not None:
            if "height" in metadata:
                self.metadata["height"] = {}
            if "view_direction" in metadata:
                self.metadata["view_direction"] = {}

        self.seed = seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

        self.bool_train = bool_train
        self.bool_binary = bool_binary

        self.images = {}
        self.labels = {}

        for elem in image_ids:

            img_path = image_folder + "/" + elem + ".tif"
            label_path = label_folder + "/" + elem + ".tif"

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                print("load {}".format(elem))
                img = Image.open(img_path)
                img = np.array(img)

                label = Image.open(label_path)
                label = np.array(label)

            if self.borders is not None:
                image_id = elem

                top = self.borders[image_id]["top"]
                bottom = self.borders[image_id]["bottom"]
                left = self.borders[image_id]["left"]
                right = self.borders[image_id]["right"]

                left = int(left + self.edge)
                right = int(right - self.edge)
                top = int(top + self.edge)
                bottom = int(bottom - self.edge)

                img = img[top:bottom, left:right]
                label = label[top:bottom, left:right]

            if img.shape != label.shape:
                print("ERROR: different shapes for Image {} and label {} at {}".format(img.shape, label.shape, elem))
                continue

            self.images[elem] = img
            self.labels[elem] = label

            # get metadata
            if "height" in self.metadata:
                sql_string = "SELECT altitude from Images where image_id='" + elem + "'"
                data = self.conn.get_data(sql_string)
                data = data.iloc[0]["altitude"]
                self.metadata["height"][elem] = data
            if "view_direction" in self.metadata:
                sql_string = "SELECT view_direction from Images where image_id='" + elem + "'"
                data = self.conn.get_data(sql_string)
                data = data.iloc[0]["view_direction"]
                if data == "V":
                    self.metadata["view_direction"][elem] = 0
                else:
                    self.metadata["view_direction"][elem] = 1

        print("---------")



    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        key = self.image_ids[idx]

        img = self.images[key]
        label = self.labels[key]

        if self.bool_binary:
            label[label <= 3] = 1
            label[label > 3] = 0
        else:
            label = label - 1

        if self.bool_train:
            img, label = self.transform_tr(img, label)
        else:
            img, label = self.transform_val(img, label)

        if "view_direction" in self.metadata:
            view_direction_channel = torch.from_numpy(np.full(img.shape, self.metadata["view_direction"][key]))

            img = torch.cat((img, view_direction_channel), axis=0)

        return img, label

    def get_weights(self, nr_labels):

        unique_counts = []
        for i in range(nr_labels):
            unique_counts.append(0)

        total = 0
        for key, elem in self.labels.items():

            unique, counts = np.unique(elem, return_counts=True)

            for i, el in enumerate(unique):
                unique_counts[el - 1] = unique_counts[el - 1] + counts[i]
                total = total + counts[i]

        percentages = []
        for elem in unique_counts:
            percentage = round((elem / total) * 100, 2)
            percentages.append(percentage)

        weights = []
        m = min(i for i in unique_counts if i > 0)
        for elem in unique_counts:
            if elem == 0:
                weight = 1
            else:
                weight = 1 / (elem / m)
            weights.append(weight)

        print(weights)

        return weights

    def get_labels(self):

        image_classes = {}

        for key, value in self.labels.items():

            pixel_classes = np.unique(value)

            image_classes[key] = pixel_classes.tolist()

        return image_classes

    def transform_tr(self, img, label, ignore=None):

        if self.aug_type == "resized":
            aug_init = album.Compose([
                album.Resize(self.img_size,self.img_size, interpolation=cv2.INTER_NEAREST),
            ])
        elif self.aug_type == "cropped":
            aug_init = album.Compose([
                Croppo(crop_size=self.img_size, input_lb=label, rc_type=self.crop_method, ignore=[6]),
            ])


        augmented = aug_init(image=img, mask=label)
        img_aug = augmented["image"]
        label_aug = augmented["mask"]

        if "flipping" in self.augmentations:
            aug_flip = album.Compose([
                album.VerticalFlip(p=0.5),
                album.HorizontalFlip(p=0.5)
                ])

            augmented = aug_flip(image=img_aug, mask=label_aug)
            img_aug = augmented["image"]
            label_aug = augmented["mask"]

        if "rotation" in self.augmentations:
            aug_rot = album.Compose([
                album.RandomRotate90(p=0.5)
            ])

            augmented = aug_rot(image=img_aug, mask=label_aug)
            img_aug = augmented["image"]
            label_aug = augmented["mask"]

        if "brightness" in self.augmentations:
            aug_br = album.Compose([
                Brighto(val=20, p=0.5)
            ])

            augmented = aug_br(image=img_aug, mask=label_aug)
            img_aug = augmented["image"]
            label_aug = augmented["mask"]

        if "noise" in self.augmentations:
            aug_noise = album.Compose([
                album.GaussNoise(var_limit=(10,50),p=0.5),
            ])

            augmented = aug_noise(image=img_aug, mask=label_aug)
            img_aug = augmented["image"]
            label_aug = augmented["mask"]

        if "normalize" in self.augmentations:
            aug_norm = album.Compose([
                Normo()
            ])

            augmented = aug_norm(image=img_aug, mask=label_aug)
            img_aug = augmented["image"]
            label_aug = augmented["mask"]

        aug_final = album.Compose([
            ToTensorV2()
        ])

        augmented = aug_final(image=img_aug, mask=label_aug)
        img_aug = augmented["image"]
        label_aug = augmented["mask"]

        img_aug = img_aug.float()
        label_aug = label_aug.to(int)

        return img_aug, label_aug

    def transform_val(self, img, label):

        aug_resize = album.Compose([album.Resize(self.img_size,self.img_size, interpolation=cv2.INTER_NEAREST)])
        aug_tensor = album.Compose([ToTensorV2()])

        resized = aug_resize(image=img, mask=label)
        img = resized["image"]
        label = resized["mask"]

        augmented = aug_tensor(image=img, mask=label)

        img_aug = augmented["image"]
        label_aug = augmented["mask"]

        img_aug = img_aug.float()
        label_aug = label_aug.to(int)

        return img_aug, label_aug


class Brighto(album.ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5, val=50):
        super(Brighto, self).__init__(always_apply, p)
        self.val = val

    def apply(self, img,**params) -> np.ndarray:

        img_c = copy.deepcopy(img)

        if random.uniform(0, 1) > self.p:

            brightness_change = random.randint(-self.val,self.val)

            img_c = img_c + brightness_change
            img_c[img_c<0] = 0
            img_c[img_c>255] = 255

        return img_c


class Normo(album.ImageOnlyTransform):

    def __init__(self, always_apply=True):
        super(Normo, self).__init__(always_apply)

    def apply(self, img, **params) -> np.ndarray:

        img_c = copy.deepcopy(img)

        img_c = img_c / 128 - 1

        return img_c


class Croppo(album.DualTransform):

    def __init__(self, always_apply=True, input_lb=None, max_size=None, crop_size=512, rc_type="random", ignore=[]):
        super(Croppo, self).__init__(always_apply)
        self.crop_size = crop_size
        self.rc_type = rc_type
        self.ignore = ignore

        img_c = copy.deepcopy(input_lb)

        if max_size is None:
            img_small = img_c[int(self.crop_size/2):img_c.shape[0]-int(self.crop_size/2),
                        int(self.crop_size/2):img_c.shape[1]-int(self.crop_size/2)]
        else:
            img_small = img_c[int(self.crop_size/2):max_size[0]-int(self.crop_size/2),
                        int(self.crop_size/2):max_size[1]-int(self.crop_size/2)]

        # get unique values
        unique, counts = np.unique(img_small, return_counts=True)
        to_remove = []
        for elem in self.ignore:
            idx_to_remove = np.where(unique == elem)[0]
            if len(idx_to_remove) > 0:
                to_remove.append(idx_to_remove[0])

        unique = np.delete(unique, to_remove)
        counts = np.delete(counts, to_remove)

        # count number of pixels
        sum = img_small.shape[0] * img_small.shape[1]

        # get weights
        weights = counts/sum

        # select the type
        if self.rc_type == "weighted":

            # normalize weights
            probs = weights/np.sum(weights)

            #select based on weights
            choice = unique[np.random.choice(len(unique), 1, p=probs)[0]]

        elif self.rc_type == "inverse":

            # inverse weights
            ln = 1/weights

            # normalize inversed weights
            in_norm = ln/np.sum(ln)

            # select based on inversed weights
            choice = unique[np.random.choice(len(unique), 1, p=in_norm)[0]]

        elif self.rc_type == "random":
            choice = unique[np.random.choice(len(unique),1)[0]]

        elif self.rc_type == "equally":

            # get number of classes
            n_classes = len(unique)

            #get probs
            probs = [1 / n_classes] * n_classes

            # select based on inversed probs
            choice = unique[np.random.choice(len(unique), 1, p=probs)[0]]

        indices = np.where(img_small == choice)
        indices = np.array(indices).T

        idx_row = np.random.choice(indices.shape[0], 1)[0]
        coords = indices[idx_row,:]

        y = coords[0]
        x = coords[1]

        min_y = y - int(self.crop_size/2)
        max_y = y + int(self.crop_size/2)

        min_x = x - int(self.crop_size/2)
        max_x = x + int(self.crop_size/2)

        # add again because we removed it before
        self.min_y = min_y + int(self.crop_size/2)
        self.max_y = max_y + int(self.crop_size/2)
        self.min_x = min_x + int(self.crop_size/2)
        self.max_x = max_x + int(self.crop_size/2)


    def apply(self, img, **params) -> np.ndarray:

        img_c = copy.deepcopy(img)

        #print("img", self.min_y, self.max_y, self.min_x, self.max_x)
        cropped = img_c[self.min_y:self.max_y, self.min_x:self.max_x]
        #print(cropped.shape)

        return cropped

    def apply_to_mask(self, img, **params):

        img_c = copy.deepcopy(img)

        #print("label", self.min_y, self.max_y, self.min_x, self.max_x)
        cropped = img_c[self.min_y:self.max_y, self.min_x:self.max_x]
        #print(cropped.shape)

        return cropped
