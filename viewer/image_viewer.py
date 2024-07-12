import copy
import cv2
import os
import numpy as np
import random

from PIL import Image, ImageTk
from tkinter import Tk, Label, Button

import src.load.load_image as li

base_fld = "/media/fdahle/beb5a64a-5335-424a-8f3c-779527060523/ATM/"
img_path = base_fld + "data_1/aerial/TMA/downloaded/"
#img_path = "/data/ATM/data_1/sfm/projects/EGU/images_orig"
input_img_type = "tif"
img_size = (800, 800)

view_directions = ["V", "R", "L"]

input_ids = ["CA196232V0035", "CA194632V0227", "CA184432V0180", "CA196232V0009", "CA194832V0074",
             "CA165231L0012", "CA194832V0066", "CA194733R0181","CA194633R0222", "CA173433R0226",
             "CA214732V0015", "CA196731L0080", "CA194732V0128", "CA164431L0040"]
input_ids = ["CA199432V0208"]
input_ids = []
input_type = "image"
#input_ids = [1801]  # ['CA172031L0258']
#input_type = "flight_path"

shuffle = False
debug_mode = False


class ImageViewer:

    def __init__(self, image_path, input_ids, input_type="image", img_type="tif",
                 view_directions=None, shuffle=False,
                 size=(800, 800), debug_mode=False):

        # params for loading
        self.image_path = image_path
        self.input_ids = input_ids
        self.input_type = input_type
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
        self.root.title("Image_Viewer")
        self.root.geometry(f"{self.size[0]}x{self.size[1]+100}")

        # gui elements
        self.img_container = Label()
        self.img_caption = Label()

        # if we have a flight path get all images from this flight path
        if self.input_type == "flight_path":
            prefixes = ["CA" + str(value) for value in self.input_ids]

            print(prefixes)

            # Get all files in the directory
            all_files = os.listdir(self.image_path)

            # Filter the files based on the prefixes
            self.input_ids = [os.path.splitext(f)[0] for f in all_files if any(f.startswith(prefix) for prefix in prefixes)]

        print(len(self.input_ids))
        print(self.input_type)

        # load input_ids
        if self.input_ids is None or len(self.input_ids) == 0:

            self.input_ids = []

            # iterate all files in folder
            for file in os.listdir(img_path):

                # only select tif files
                if file.endswith(".tif"):
                    # remove the .tif
                    self.input_ids.append(file.split(".")[0])

        # filter for view direction
        if view_directions is not None:
            if len(view_directions) == 0:
                view_directions = ["L", "V", "R"]
            temp = copy.deepcopy(self.input_ids)
            self.input_ids = []
            for elem in temp:
                if any(x in elem for x in view_directions):
                    self.input_ids.append(elem)

        if shuffle:
            random.shuffle(self.input_ids)
        else:
            self.input_ids.sort()

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

        # Label to show the pixel value
        self.pixel_value_label = Label(self.root, text="Value: ")
        self.pixel_value_label.grid(row=7, column=1)

        # add mouse moving event
        self.img_container.bind('<Motion>', self.on_mouse_move)

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
            self.img_pos = len(self.input_ids) - 1
        elif self.img_pos == len(self.input_ids):
            self.img_pos = 0

        self.image_id = self.input_ids[self.img_pos]

        if self.debug_mode:
            print(f"Load image '{self.image_id}'")

        if self.image_id not in self.img_dict:

            np_arr = li.load_image(self.image_id,
                                               image_type=self.img_type,
                                               image_path=self.image_path)

            if np_arr is None:
                np_arr = np.ones(self.size)

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

    # callback method for mouse move event
    def on_mouse_move(self, event):
        # Convert event coordinates to image coordinates
        x, y = event.x, event.y

        # Check if the coordinates are within the image bounds
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            # Retrieve the pixel value at (x, y)
            pixel_value = self.img_dict[self.image_id][y, x]  # Adjust indexing if necessary

            # Update the label text with the new pixel value
            self.pixel_value_label.config(text=f"Value: {pixel_value}")
        else:
            # If out of bounds, clear the label or display a default message
            self.pixel_value_label.config(text="Value: Out of bounds")


if __name__ == "__main__":

    viewer = ImageViewer(img_path, input_ids, input_type,
                         size=img_size,
                         view_directions=view_directions,
                         shuffle=shuffle,
                         img_type=input_img_type)
