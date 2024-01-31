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
img_size = (600, 600)
tweak_val = 100

view_directions = None
input_ids = ["CA183531L0079"]
shuffle = False
debug_mode = True


class SubsetFidViewer:

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
        self.image_id = None
        self.img_pos = 0  # position in image_ids we have loaded, start with 0

        # keep temporary copies
        self.img_dict = {}
        self.subset_north_dict = {}
        self.subset_south_dict = {}
        self.subset_west_dict = {}
        self.subset_east_dict = {}

        self.subset_north_west_dict = {}
        self.subset_north_east_dict = {}
        self.subset_south_west_dict = {}
        self.subset_south_east_dict = {}

        # create window
        self.root = Tk()
        self.root.title("Subset_Fid_Viewer")
        self.root.geometry(f"{self.size[0] + 300}x{self.size[1] + 400}")

        # gui elements
        self.img_container = Label()
        self.img_caption = Label()
        self.subset_north_container = Label()
        self.subset_south_container = Label()
        self.subset_west_container = Label()
        self.subset_east_container = Label()

        self.subset_north_west_container = Label()
        self.subset_north_east_container = Label()
        self.subset_south_west_container = Label()
        self.subset_south_east_container = Label()

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

        if shuffle:
            random.shuffle(self.image_ids)

        if debug_mode:
            print(f"View images from {self.image_path}")

        # set position of image name and container
        self.img_caption.grid(row=10, column=2, columnspan=3)
        self.img_container.grid(row=3, column=1, rowspan=5, columnspan=5)

        # set position of the subsets
        self.subset_north_container.grid(row=0, column=2, rowspan=1, columnspan=3)
        self.subset_south_container.grid(row=8, column=2, rowspan=1, columnspan=3)
        self.subset_west_container.grid(row=4, column=0, rowspan=3, columnspan=1)
        self.subset_east_container.grid(row=4, column=6, rowspan=3, columnspan=1)

        self.subset_north_west_container.grid(row=0, column=0, rowspan=1, columnspan=1)
        self.subset_north_east_container.grid(row=8, column=0, rowspan=1, columnspan=1)
        self.subset_south_west_container.grid(row=8, column=6, rowspan=1, columnspan=1)
        self.subset_south_east_container.grid(row=0, column=6, rowspan=1, columnspan=1)

        # We will have three button back ,forward and exit
        button_back = Button(self.root, text="Back", command=lambda: self.load_image("back"))
        button_exit = Button(self.root, text="Exit", command=self.root.quit)
        button_forward = Button(self.root, text="Forward", command=lambda: self.load_image("forward"))

        # grid function is for placing the buttons in the frame
        button_back.grid(row=9, column=1)
        button_exit.grid(row=9, column=3)
        button_forward.grid(row=9, column=5)

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

            # get subset_data
            sql_string = f"SELECT * FROM images_fid_points WHERE image_id='{self.image_id}'"
            data = ctd.get_data_from_db(sql_string)
            data = data.iloc[0]

            # get subsets
            np_arr_subsets = {}
            for key in ["n", "s", "e", "w"]:
                try:
                    min_x = int(data[f'subset_{key}_x'])
                    max_x = int(data[f'subset_{key}_x']) + int(data['subset_width'])
                    min_y = int(data[f'subset_{key}_y'])
                    max_y = int(data[f'subset_{key}_y']) + int(data['subset_height'])

                    if key == "n":
                        min_y = min_y - tweak_val
                    elif key == "s":
                        max_y = max_y + tweak_val
                    elif key == "w":
                        min_x = min_x - tweak_val
                    elif key == "e":
                        max_x = max_x + tweak_val

                    min_y = max(min_y, 0)
                    min_x = max(min_x, 0)
                    max_y = min(max_y, np_arr.shape[0])
                    max_x = min(max_x, np_arr.shape[1])

                    np_arr_subset = np_arr[min_y:max_y, min_x:max_x]
                    if len(np_arr_subset) == 0:
                        raise ValueError("Subset is empty")
                except (Exception,):
                    np_arr_subset = np.ones([int(data['subset_height']), int(data['subset_width'])])
                np_arr_subsets[key] = np_arr_subset

            # draw fid points
            for i in range(8):
                try:
                    fid_x = int(data[f'fid_mark_{i}_x'])
                    fid_y = int(data[f'fid_mark_{i}_y'])
                    cv2.circle(np_arr, (fid_x, fid_y), 5, (255, 0, 0), 3)
                except (Exception,):
                    pass

            np_arr_north_west = np.ones([int(data['subset_height']), int(data['subset_width'])])
            np_arr_north_east = np.ones([int(data['subset_height']), int(data['subset_width'])])
            np_arr_south_west = np.ones([int(data['subset_height']), int(data['subset_width'])])
            np_arr_south_east = np.ones([int(data['subset_height']), int(data['subset_width'])])

            np_arr = cv2.resize(np_arr, self.size)
            np_arr_north = cv2.resize(np_arr_subsets["n"], (150, 150))
            np_arr_south = cv2.resize(np_arr_subsets["s"], (150, 150))
            np_arr_west = cv2.resize(np_arr_subsets["w"], (150, 150))
            np_arr_east = cv2.resize(np_arr_subsets["e"], (150, 150))

            np_arr_north_west = cv2.resize(np_arr_north_west, (150, 150))
            np_arr_north_east = cv2.resize(np_arr_north_east, (150, 150))
            np_arr_south_west = cv2.resize(np_arr_south_west, (150, 150))
            np_arr_south_east = cv2.resize(np_arr_south_east, (150, 150))

            # save temp copy
            self.img_dict[self.image_id] = np_arr
            self.subset_north_dict[self.image_id] = np_arr_north
            self.subset_south_dict[self.image_id] = np_arr_south
            self.subset_west_dict[self.image_id] = np_arr_west
            self.subset_east_dict[self.image_id] = np_arr_east

            self.subset_north_west_dict[self.image_id] = np_arr_north_west
            self.subset_north_east_dict[self.image_id] = np_arr_north_east
            self.subset_south_west_dict[self.image_id] = np_arr_south_west
            self.subset_south_east_dict[self.image_id] = np_arr_south_east

        arr = self.img_dict[self.image_id]
        arr_north = self.subset_north_dict[self.image_id]
        arr_south = self.subset_south_dict[self.image_id]
        arr_west = self.subset_west_dict[self.image_id]
        arr_east = self.subset_east_dict[self.image_id]

        arr_north_west = self.subset_north_west_dict[self.image_id]
        arr_north_east = self.subset_north_east_dict[self.image_id]
        arr_south_west = self.subset_south_west_dict[self.image_id]
        arr_south_east = self.subset_south_east_dict[self.image_id]

        photo = ImageTk.PhotoImage(image=Image.fromarray(arr))
        self.img_container.configure(image=photo)
        self.img_container.image = photo

        photo_north = ImageTk.PhotoImage(image=Image.fromarray(arr_north))
        self.subset_north_container.configure(image=photo_north)
        self.subset_north_container.image = photo_north
        photo_south = ImageTk.PhotoImage(image=Image.fromarray(arr_south))
        self.subset_south_container.configure(image=photo_south)
        self.subset_south_container.image = photo_south
        photo_west = ImageTk.PhotoImage(image=Image.fromarray(arr_west))
        self.subset_west_container.configure(image=photo_west)
        self.subset_west_container.image = photo_west
        photo_east = ImageTk.PhotoImage(image=Image.fromarray(arr_east))
        self.subset_east_container.configure(image=photo_east)
        self.subset_east_container.image = photo_east
        photo_north_west = ImageTk.PhotoImage(image=Image.fromarray(arr_north_west))
        self.subset_north_west_container.configure(image=photo_north_west)
        self.subset_north_west_container.image = photo_north_west
        photo_north_east = ImageTk.PhotoImage(image=Image.fromarray(arr_north_east))
        self.subset_north_east_container.configure(image=photo_north_east)
        self.subset_north_east_container.image = photo_north_east
        photo_south_west = ImageTk.PhotoImage(image=Image.fromarray(arr_south_west))
        self.subset_south_west_container.configure(image=photo_south_west)
        self.subset_south_west_container.image = photo_south_west
        photo_south_east = ImageTk.PhotoImage(image=Image.fromarray(arr_south_east))
        self.subset_south_east_container.configure(image=photo_south_east)
        self.subset_south_east_container.image = photo_south_east

        self.img_caption.configure(text=self.image_id)

    def change_view(self, change_type):

        if change_type == "rotate_right":
            self.img_dict[self.image_id] = np.rot90(self.img_dict[self.image_id], 1)
        elif change_type == "rotate_left":
            self.img_dict[self.image_id] = np.rot90(self.img_dict[self.image_id], 3)

        self.load_image()


if __name__ == "__main__":
    viewer = SubsetFidViewer(img_path, input_ids,
                             size=img_size,
                             view_directions=view_directions,
                             shuffle=shuffle,
                             img_type=input_img_type)
