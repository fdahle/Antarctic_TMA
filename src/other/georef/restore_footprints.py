"""
The shapes in the overview shapefile are not consistently real squares, but sometimes
have small deviations. This function recalculate the shapes again
"""

import glob
import os

import src.export.export_geometry as eg
import src.georef.snippets.convert_image_to_footprint as ctf
import src.load.load_image as li
import src.load.load_shape_data as lsd
import src.load.load_transform as lt

from tqdm import tqdm

image_id = None
img_type = "sat"
use_attributes_from_shp = True

# get all image ids from folder if image_id is not provided
if image_id is None:
    path_fld = f"/data/ATM/data_1/georef/{img_type}"
    pattern = os.path.join(path_fld, '*.tif')
    tif_files = glob.glob(pattern)
    image_ids = [os.path.basename(file)[:-4] for file in tif_files]
else:
    image_ids = [image_id]

if use_attributes_from_shp:
    # get the attributes from the shapefile
    path_shp_file = f"/data/ATM/data_1/georef/{img_type}.shp"
    shape_data = lsd.load_shape_data(path_shp_file)

path_new_shp_file = f"/data/ATM/data_1/georef/{img_type}_new.shp"

for img_id in tqdm(image_ids):

    # load the image
    image = li.load_image(img_id)

    # load the transform
    transform_path = f"/data/ATM/data_1/georef/{img_type}/{img_id}_transform.txt"
    transform = lt.load_transform(transform_path)

    # convert the image to a footprint
    footprint = ctf.convert_image_to_footprint(image, transform)

    # get the attributes from the shapefile for that image id
    if use_attributes_from_shp:

        # check if image_id is in shape_data
        if img_id not in shape_data["image_id"].values:
            continue

        attributes = shape_data[shape_data["image_id"] == img_id].to_dict(orient="records")[0]

        # remove geometry from attributes
        attributes.pop("geometry")

    else:
        attributes = {
            "image_id": img_id,
        }

    eg.export_geometry(footprint, path_new_shp_file,
                       attributes=attributes, key_field="image_id",
                       attach=True)
