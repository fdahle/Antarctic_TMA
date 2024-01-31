import copy
import cv2
import os
import numpy as np
import random

from PIL import Image, ImageTk
from tkinter import Tk, Label, Button

import base.load_image_from_file as liff

img_path = "/data_1/ATM/data_1/aerial/TMA/downloaded/"
segmented_path = "/data_1/ATM/data_1/aerial/TMA/segmented/supervised"
input_img_type = "tif"
img_size = (400, 400)

view_directions = ["V"]
input_ids = ["CA135433R0350"]
shuffle = False
debug_mode = True


class SegmentedViewer:

    def __init__(self, image_path, segmented_path, image_ids=None, img_type="tif",
                 view_directions=None, shuffle=False,
                 size=(800, 800), debug_mode=False):

        # params for loading
        self.image_path = image_path
        self.segmented_path = segmented_path
        self.image_ids = image_ids
        self.img_type = img_type
        self.view_directions = view_directions
        self.shuffle = shuffle
        self.size = size

        self.debug_mode = debug_mode

        # params during run-time
        self.img_dict = {}  # keep temporary copies
        self.segmented_dict = {}  # keep temporary copies
        self.image_id = None
        self.img_pos = 0  # position in image_ids we have loaded, start with 0

        # create window
        self.root = Tk()
        self.root.title("Segmented_Viewer")
        self.root.geometry(f"{(self.size[0] + 75) * 2}x{self.size[1] + 100}")

        # gui elements
        self.img_container = Label()
        self.segmented_container = Label()
        self.img_caption = Label()

        # load image_ids
        if self.image_ids is None or len(self.image_ids) == 0:

            self.image_ids = []
            self.segmented_ids = []

            # iterate all files in folder
            for file in os.listdir(img_path):

                # only select tif files
                if file.endswith(".tif"):
                    # remove the .tif
                    self.image_ids.append(file.split(".")[0])

            for file in os.listdir(segmented_path):

                # only select tif files
                if file.endswith(".tif"):
                    # remove the .tif
                    self.segmented_ids.append(file.split(".")[0])

            # filter for same values
            self.image_ids = set(self.image_ids) & set(self.segmented_ids)

        # filter for view direction (when no input ids are given)
        if view_directions is not None and len(input_ids) == 0:
            temp = copy.deepcopy(self.image_ids)
            self.image_ids = []
            for elem in temp:
                if any(x in elem for x in view_directions):
                    self.image_ids.append(elem)

        if len(self.image_ids) == 0:
            print("No images could be found!")
            exit()

        if shuffle:
            random.shuffle(self.image_ids)

        if debug_mode:
            print(f"View images from {self.image_path}")

        # set position of image name and container
        self.img_caption.grid(row=0, column=0, columnspan=3)
        self.img_container.grid(row=2, column=0, rowspan=4, columnspan=2)
        self.segmented_container.grid(row=2, column=3, rowspan=4, columnspan=2)

        # We will have two buttons rotate left and rotate right
        button_rotate_left = Button(self.root, text="Rotate left",
                                    command=lambda: self.change_view("rotate_left"))
        button_rotate_right = Button(self.root, text="Rotate right",
                                     command=lambda: self.change_view("rotate_right"))

        # grid function is for placing the buttons in the frame
        button_rotate_left.grid(row=1, column=0)
        button_rotate_right.grid(row=1, column=2)

        # We will have three button back ,forward and exit
        button_back = Button(self.root, text="Back", command=lambda: self.load_image("back"))
        button_exit = Button(self.root, text="Exit", command=self.root.quit)
        button_forward = Button(self.root, text="Forward", command=lambda: self.load_image("forward"))

        # grid function is for placing the buttons in the frame
        button_back.grid(row=6, column=0)
        button_exit.grid(row=6, column=1)
        button_forward.grid(row=6, column=2)

        # load image
        self.load_image()

        # show the windows
        self.root.mainloop()

    # load the image
    def load_image(self, direction=None):

        if direction == "forward":
            self.img_pos = self.img_pos + 1
        elif direction == "back":
            self.img_pos = self.img_pos - 1

        if self.img_pos < 0:
            self.img_pos = len(self.image_ids) - 1
        elif self.img_pos == len(self.image_ids):
            self.img_pos = 0

        self.image_id = self.image_ids[self.img_pos]

        if self.debug_mode:
            print(f"Load image '{self.image_id}'")

        if self.image_id not in self.img_dict:

            np_arr = liff.load_image_from_file(self.image_id,
                                               image_type=self.img_type,
                                               image_path=self.image_path)
            np_segmented_arr = liff.load_image_from_file(self.image_id,
                                                         image_type=self.img_type,
                                                         image_path=self.segmented_path)

            if np_arr is None:
                np_arr = np.ones(self.size)
            if np_segmented_arr is None:
                np_segmented_arr = np.ones(self.size)

            color_arr = [
                (102, 255, 0),  # for no data, 0
                (150, 149, 158),  # ice, 1
                (230, 230, 235),  # snow, 2
                (46, 45, 46),  # rocks, 3
                (7, 29, 232),  # water, 4
                (25, 227, 224),  # clouds, 5
                (186, 39, 32),  # sky, 6
                (224, 7, 224)  # unknown, 7
            ]

            # apply color to the image
            np_segmented_arr = np.take(color_arr, np_segmented_arr, axis=0).astype(np.uint8)

            np_arr = cv2.resize(np_arr, self.size)
            np_segmented_arr = cv2.resize(np_segmented_arr, self.size)

            # save temp copy
            self.img_dict[self.image_id] = np_arr
            self.segmented_dict[self.image_id] = np_segmented_arr

        arr = self.img_dict[self.image_id]
        arr_segmented = self.segmented_dict[self.image_id]

        photo = ImageTk.PhotoImage(image=Image.fromarray(arr))
        photo_segmented = ImageTk.PhotoImage(image=Image.fromarray(arr_segmented))

        self.img_container.configure(image=photo)
        self.segmented_container.configure(image=photo_segmented)

        self.img_container.image = photo
        self.segmented_container.image = photo_segmented

        self.img_caption.configure(text=self.image_id)

    def change_view(self, change_type):

        if change_type == "rotate_right":
            self.img_dict[self.image_id] = np.rot90(self.img_dict[self.image_id], 1)
            self.segmented_dict[self.image_id] = np.rot90(self.segmented_dict[self.image_id], 1)
        elif change_type == "rotate_left":
            self.img_dict[self.image_id] = np.rot90(self.img_dict[self.image_id], 3)
            self.segmented_dict[self.image_id] = np.rot90(self.segmented_dict[self.image_id], 3)

        self.load_image()


if __name__ == "__main__":
    viewer = SegmentedViewer(img_path, segmented_path, input_ids,
                             size=img_size,
                             view_directions=view_directions,
                             shuffle=shuffle,
                             img_type=input_img_type)
