import copy
import cv2
import os
import numpy as np
import random
import threading  # for background preloading

from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, Frame
from tqdm import tqdm

import src.load.load_image as li

# Global parameters for paths, image type, etc.
base_fld = "/data/ATM/"
img_path = base_fld + "data_1/aerial/TMA/downloaded/"
img_path_backup = "/media/fdahle/d3f2d1f5-52c3-4464-9142-3ad7ab1ec06d/data_1/aerial/TMA/downloaded"
input_img_type = "tif"

# New naming:
window_size = (1800, 800)   # overall window size and image resize dimensions
crop_size = (150, 150)      # the area to extract from the full image
subset_size = (320, 320)    # the size to display each cropped subset

view_directions = ["V"]

# Example flight path IDs
input_ids = [1822]
input_type = "flight_path"

shuffle = False
debug_mode = False
load_all = True

class FidMarkViewer:
    def __init__(self, image_path, input_ids, input_type="image", img_type="tif",
                 view_directions=None, shuffle=False, window_size=(1500,800),
                 debug_mode=False, crop_size=(150,150), subset_size=None):
        # If subset_size is not provided, default to crop_size.
        if subset_size is None:
            subset_size = crop_size

        # Parameters for loading images
        self.image_path = image_path
        self.input_ids = input_ids
        self.input_type = input_type
        self.img_type = img_type
        self.view_directions = view_directions
        self.shuffle = shuffle
        self.window_size = window_size
        self.crop_size = crop_size
        self.subset_size = subset_size
        self.preload_count = 5
        self.debug_mode = debug_mode

        # Dictionary to cache loaded images
        self.img_dict = {}
        self.image_id = None
        self.img_pos = 0  # current position in input_ids

        # Create main window
        self.root = Tk()
        self.root.title("Fiducial Mark Viewer")
        self.root.geometry(f"{self.window_size[0]}x{self.window_size[1] + 100}")

        # If a flight path was provided, update input_ids based on file prefixes.
        if self.input_type == "flight_path":
            prefixes = ["CA" + str(value) for value in self.input_ids]
            all_files = os.listdir(self.image_path)
            all_files_backup = os.listdir(img_path_backup)
            all_files = list(set(all_files + all_files_backup))
            self.input_ids = [os.path.splitext(f)[0] for f in all_files if any(f.startswith(prefix) for prefix in prefixes)]

        # If no input_ids were provided, load all tif files from the folder.
        if not self.input_ids:
            self.input_ids = []
            for file in os.listdir(self.image_path):
                if file.endswith(".tif"):
                    self.input_ids.append(os.path.splitext(file)[0])

        # Filter for view directions if provided.
        if self.view_directions is not None:
            if len(self.view_directions) == 0:
                self.view_directions = ["L", "V", "R"]
            temp = copy.deepcopy(self.input_ids)
            self.input_ids = []
            for elem in temp:
                if any(x in elem for x in self.view_directions):
                    self.input_ids.append(elem)

        if self.shuffle:
            random.shuffle(self.input_ids)
        else:
            self.input_ids.sort()
        print(len(self.input_ids), "images found.")

        if self.debug_mode:
            print(f"Viewing images from {self.image_path}")

        # GUI elements
        self.img_caption = Label(self.root)
        self.img_caption.grid(row=0, column=0, columnspan=5)

        # Instead of one large image container, create a frame for the 5 thumbnails.
        self.thumb_frame = Frame(self.root)
        self.thumb_frame.grid(row=2, column=0, columnspan=5, padx=5, pady=5)

        # Buttons for rotation
        button_rotate_left = Button(self.root, text="Rotate left",
                                    command=lambda: self.change_view("rotate_left"))
        button_rotate_right = Button(self.root, text="Rotate right",
                                     command=lambda: self.change_view("rotate_right"))
        button_rotate_left.grid(row=1, column=0)
        button_rotate_right.grid(row=1, column=4)

        # Navigation buttons: Back, Forward, and Exit
        button_back = Button(self.root, text="Back", command=lambda: self.load_image("back"))
        button_exit = Button(self.root, text="Exit", command=self.root.quit)
        button_forward = Button(self.root, text="Forward", command=lambda: self.load_image("forward"))
        button_back.grid(row=6, column=0)
        button_exit.grid(row=6, column=2)
        button_forward.grid(row=6, column=4)

        # Label to show the pixel value
        self.pixel_value_label = Label(self.root, text="Value: ")
        self.pixel_value_label.grid(row=7, column=2)

        # Bind mouse motion events (to the thumbnail frame)
        self.thumb_frame.bind('<Motion>', self.on_mouse_move)

        # Bind arrow keys for navigation
        self.root.bind("<Left>", self.handle_left_arrow)
        self.root.bind("<Right>", self.handle_right_arrow)

        # Optionally preload all images
        if load_all:
            print("Loading all images...")
            self.preload_all_images()

        # Load the first image
        self.load_image()

        self.root.mainloop()

    def preload_all_images(self):
        # Preload all images with a progress bar.
        for image_id in tqdm(self.input_ids, desc="Loading all images"):
            if image_id not in self.img_dict:
                print(image_id)
                np_arr = li.load_image(image_id,
                                       image_type=self.img_type,
                                       image_path=self.image_path,
                                       catch=True)
                if np_arr is None:
                    np_arr = np.ones(self.window_size)
                np_arr = cv2.resize(np_arr, self.window_size)
                self.img_dict[image_id] = np_arr

    def handle_left_arrow(self, event):
        self.load_image("back")

    def handle_right_arrow(self, event):
        self.load_image("forward")

    def load_image(self, direction=None):
        # Update image index based on navigation.
        if direction == "forward":
            self.img_pos = (self.img_pos + 1) % len(self.input_ids)
        elif direction == "back":
            self.img_pos = (self.img_pos - 1) % len(self.input_ids)

        self.image_id = self.input_ids[self.img_pos]
        if self.debug_mode:
            print(f"Loading image '{self.image_id}'")

        # Load the image if it isn’t cached.
        if self.image_id not in self.img_dict:
            np_arr = li.load_image(self.image_id,
                                   image_type=self.img_type,
                                   image_path=self.image_path)
            if np_arr is None:
                np_arr = np.ones(self.window_size)
            np_arr = cv2.resize(np_arr, self.window_size)
            self.img_dict[self.image_id] = np_arr

        img = self.img_dict[self.image_id]
        self.img_caption.configure(text=self.image_id)

        # Always display the 5 crops in the thumbnail frame.
        self.display_crops(img)

        # Start background preloading of adjacent images.
        preload_direction = direction if direction is not None else "forward"
        preload_thread = threading.Thread(target=self.preload_images, args=(preload_direction, self.preload_count))
        preload_thread.daemon = True
        preload_thread.start()

    def preload_images(self, direction, count):
        if direction == "forward":
            indices = [(self.img_pos + i) % len(self.input_ids) for i in range(1, count+1)]
        elif direction == "back":
            indices = [(self.img_pos - i) % len(self.input_ids) for i in range(1, count+1)]
        else:
            raise ValueError(f"Invalid direction '{direction}' for preloading images.")
        for idx in indices:
            image_id = self.input_ids[idx]
            if image_id not in self.img_dict:
                if self.debug_mode:
                    print(f"Preloading image '{image_id}'")
                np_arr = li.load_image(image_id,
                                       image_type=self.img_type,
                                       image_path=self.image_path)
                if np_arr is None:
                    np_arr = np.ones(self.window_size)
                np_arr = cv2.resize(np_arr, self.window_size)
                self.img_dict[image_id] = np_arr

    def display_crops(self, img):
        # Clear any existing thumbnails.
        for widget in self.thumb_frame.winfo_children():
            widget.destroy()

        # Compute crop boundaries based on crop_size.
        crop_w, crop_h = self.crop_size
        half_w = crop_w // 2
        half_h = crop_h // 2
        full_w, full_h = self.window_size
        center_x, center_y = full_w // 2, full_h // 2

        # Define the five crops from the full image.
        left_crop   = img[center_y - half_h : center_y + half_h, 0:crop_w]
        right_crop  = img[center_y - half_h : center_y + half_h, full_w - crop_w : full_w]
        top_crop    = img[0:crop_h, center_x - half_w : center_x + half_w]
        bottom_crop = img[full_h - crop_h : full_h, center_x - half_w : center_x + half_w]
        full_img    = img.copy()  # full image thumbnail (will be resized)

        # Resize each crop to the display (subset) size.
        crops = [
            cv2.resize(top_crop, self.subset_size),
            cv2.resize(right_crop, self.subset_size),
            cv2.resize(bottom_crop, self.subset_size),
            cv2.resize(left_crop, self.subset_size),
            cv2.resize(full_img, self.subset_size)
        ]

        # Create and place a Label for each crop in a horizontal row.
        for idx, crop in enumerate(crops):
            photo = ImageTk.PhotoImage(image=Image.fromarray(crop))
            thumb_label = Label(self.thumb_frame, image=photo)
            thumb_label.image = photo  # keep a reference to avoid garbage collection
            thumb_label.grid(row=0, column=idx, padx=5, pady=5)
            thumb_label.bind('<Motion>', self.on_mouse_move)

    def change_view(self, change_type):
        # Rotate the current image and refresh the crops.
        if change_type == "rotate_right":
            self.img_dict[self.image_id] = np.rot90(self.img_dict[self.image_id], 1)
        elif change_type == "rotate_left":
            self.img_dict[self.image_id] = np.rot90(self.img_dict[self.image_id], 3)
        self.load_image()

    def on_mouse_move(self, event):
        # For simplicity, assume each thumbnail is subset_size in dimensions.
        x, y = event.x, event.y
        subset_w, subset_h = self.subset_size
        if 0 <= x < subset_w and 0 <= y < subset_h:
            self.pixel_value_label.config(text=f"Value: ({x}, {y})")
        else:
            self.pixel_value_label.config(text="Value: Out of bounds")

def fid_mark_viewer(image_path, input_ids, input_type="image", img_type="tif",
                    view_directions=None, shuffle=False, window_size=(1500,800),
                    debug_mode=False, crop_size=(150,150), subset_size=None):
    """
    This function instantiates and runs the FidMarkViewer which always displays 5 crops.
    """
    return FidMarkViewer(image_path, input_ids, input_type, img_type,
                         view_directions, shuffle, window_size,
                         debug_mode, crop_size, subset_size)

if __name__ == "__main__":
    fid_mark_viewer(img_path, input_ids, input_type,
                    img_type=input_img_type,
                    view_directions=view_directions,
                    shuffle=shuffle,
                    window_size=window_size,
                    debug_mode=debug_mode,
                    crop_size=crop_size,
                    subset_size=subset_size)
