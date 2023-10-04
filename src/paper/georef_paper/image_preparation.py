image_id = "CA182632V0127"

import base.load_image_from_file as liff

img = liff.load_image_from_file(image_id)

import base.remove_borders as rb
img, edge_dims = rb.remove_borders(img, image_id, return_edge_dims=True)

print(edge_dims)
print(edge_dims)

# create mask and remove borders
import numpy as np
mask = np.ones_like(img)
mask[:edge_dims[2], :] = 0
mask[edge_dims[3]:, :] = 0
mask[:, :edge_dims[0]] = 0
mask[:, edge_dims[1]:] = 0

# get data for text boxes
import base.connect_to_db as ctd
sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id='{image_id}'"
data = ctd.get_data_from_db(sql_string, catch=True)
text_data = data.iloc[0]['text_bbox']

# get the text boxes from data
text_boxes = text_data.split(";")

# Convert string representation to list of tuples
coordinate_pairs = [tuple(map(lambda x: int(float(x)), coord[1:-1].split(','))) for coord in text_boxes]

# Set the regions in the array to 0
for top_left_x, top_left_y, bottom_right_x, bottom_right_y in coordinate_pairs:
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

import display.display_images as di
#di.display_images([mask])

import base.resize_image as resi
img = resi.resize_image(img, (1000, 1000))
mask = resi.resize_image(mask, (1000, 1000))

sql_string = f"SELECT * FROM images WHERE image_id='{image_id}'"
data = ctd.get_data_from_db(sql_string)
data = data.iloc[0]

azimuth = data["azimuth"]

import image_georeferencing.sub.enhance_image as eh
img = eh.enhance_image(img, scale=(0.02, 0.98))

di.display_images(img)


import base.rotate_image as ri
img = ri.rotate_image(img, azimuth)

di.display_images(img)
