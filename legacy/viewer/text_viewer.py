import copy
import cv2
import os
import numpy as np
import random

from PIL import Image, ImageTk
from tkinter import Tk, Label, Button

import base.connect_to_db as ctd
import base.load_image_from_file as liff

img_path = "/data_1/ATM/data_1/aerial/TMA/downloaded/"
input_img_type = "tif"
img_size = (800, 800)

view_directions = ["R", "V", "L"]
input_ids = []
shuffle = False
debug_mode = True


class TextViewer:

    def __init__(self, image_path, image_ids=None, img_type="tif",
                 view_directions=None, shuffle=False,
                 size=(800, 800), debug_mode=False):

        # params for loading
        self.image_path = image_path
        self.image_ids = image_ids
        self.img_type = img_type
        self.view_directions = view_directions
        self.shuffle = shuffle
        self.size = size

        self.debug_mode = debug_mode

        # params during run-time
        self.img_dict = {}  # keep temporary copies
        self.image_id = None
        self.img_pos = 0  # position in image_ids we have loaded, start with 0

        # create window
        self.root = Tk()
        self.root.title("Text_Viewer")
        self.root.geometry(f"{self.size[0]}x{self.size[1]+100}")

        # gui elements
        self.img_container = Label()
        self.img_caption = Label()

        # load image_ids
        if self.image_ids is None or len(self.image_ids) == 0:

            self.image_ids = []

            # iterate all files in folder
            for file in os.listdir(img_path):
                # only select tif files
                if file.endswith(".tif"):

                    # remove the .tif
                    self.image_ids.append(file.split(".")[0])

        # filter for view direction
        if view_directions is not None:
            temp = copy.deepcopy(self.image_ids)
            self.image_ids = []
            for elem in temp:
                if any(x in elem for x in view_directions):
                    self.image_ids.append(elem)

        if len(self.image_ids) == 0:
            print("No images are left with the combination of ids and view-direction")
            exit()

        if shuffle:
            random.shuffle(self.image_ids)

        if debug_mode:
            print(f"View images from {self.image_path}")

        # set position of image name and container
        self.img_caption.grid(row=0, column=0, columnspan=3)
        self.img_container.grid(row=2, column=0, rowspan=4, columnspan=3)

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

            # load the array
            np_arr = liff.load_image_from_file(self.image_id,
                                               image_type=self.img_type,
                                               image_path=self.image_path)

            if np_arr is None:
                np_arr = np.ones(self.size)

            # convert to rgb (to show coloured points)
            np_arr = np.stack((np_arr,) * 3, axis=-1)

            # get text
            sql_string = f"SELECT text_bbox, text_content " \
                         f"FROM images_extracted WHERE image_id='{self.image_id}'"
            data = ctd.get_data_from_db(sql_string)
            data = data.iloc[0]

            # textbox is [min_x_abs, min_y_abs, max_x_abs, max_y_abs]
            if data["text_bbox"] is not None:
                rectangles = data["text_bbox"].split(";")
                for elem in rectangles:
                    # remove brackets
                    elem = elem[1:-1]
                    points = elem.split(",")
                    start_point = (int(points[0]), int(points[1]))
                    end_point = (int(points[2]), int(points[3]))
                    print(start_point, end_point)
                    #start_point = (100, 100)
                    #end_point = (500, 500)
                    np_arr = cv2.rectangle(np_arr, start_point, end_point, (255, 0, 0), 10)

            np_arr = cv2.resize(np_arr, self.size)

            # save temp copy
            self.img_dict[self.image_id] = np_arr

        arr = self.img_dict[self.image_id]
        photo = ImageTk.PhotoImage(image=Image.fromarray(arr))
        self.img_container.configure(image=photo)
        self.img_container.image = photo

        self.img_caption.configure(text=self.image_id)

    def change_view(self, change_type):

        if change_type == "rotate_right":
            self.img_dict[self.image_id] = np.rot90(self.img_dict[self.image_id], 1)
        elif change_type == "rotate_left":
            self.img_dict[self.image_id] = np.rot90(self.img_dict[self.image_id], 3)

        self.load_image()


if __name__ == "__main__":

    viewer = TextViewer(img_path, input_ids,
                         size=img_size,
                         view_directions=view_directions,
                         shuffle=shuffle,
                         img_type=input_img_type)
