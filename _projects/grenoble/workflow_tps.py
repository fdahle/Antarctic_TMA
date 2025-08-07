project = "iceland"
img_type = "aerial_images"
processing_type = "preprocessed"
invert_mask = True

# define path to project
input_fld = f"/data/ATM/grenoble_project/{project}/input/"
work_fld = f"/data/ATM/grenoble_project/{project}/intermediate/"

# define paths for input images
input_img_fld = input_fld + img_type + "/" + processing_type + "_images/images/"
image_prefix = "F-"

import src.load.load_image as li
import src.display.display_images as di

# load the mask
mask = li.load_image(input_fld + img_type + "/" + processing_type + "_images/image_mask.tif")
if invert_mask:
    mask=1-mask

import src.base.rotate_image as ri
mask_rotated = ri.rotate_image(mask, 180)

# get all tif tiles in the input folder starting with "F_"
import glob

# get list of image paths and image names
lst_image_pth = glob.glob(input_img_fld + image_prefix + "*.tif")
lst_image_pth.sort()
lst_image_nms = [p.split("/")[-1][:-4] for p in lst_image_pth]

print(lst_image_nms)

import src.base.find_tie_points as ftp

# load the footprints
import src.load.load_shape_data as lsd

footprints = lsd.load_shape_data(input_fld + img_type + "/" + "images_footprint.shp")

# Determine which column to use
id_column = 'Image ID' if 'Image ID' in footprints.columns else 'Entity  ID'

# Build the mapping
geom_map = {row[id_column].replace('.tif', ''): row['geometry'] for _, row in footprints.iterrows()}

# Get ordered polygons
ordered_polygons = [geom_map[img] for img in lst_image_nms if img in geom_map]

# find overlapping images
import src.base.find_overlapping_images_geom as foig
overlapping_images = foig.find_overlapping_images_geom(lst_image_nms,
                                                       ordered_polygons,
                                                       min_overlap=0.1)


tpd = ftp.TiePointDetector("lightglue", min_conf=0.9,
                           catch=False, verbose=True, display=False)

import numpy as np
import os

output_fld = f"/data/ATM/grenoble_project/{project}/intermediate/{img_type}/tie_points/"

display = False

import src.base.rotate_points as rp

for key, value in overlapping_images.items():

    img1 = li.load_image(input_img_fld + key + ".tif")

    print(key, value)

    for img2_name in value:

        print("Processing tie points for:", key, img2_name)

        img2 = li.load_image(input_img_fld + img2_name + ".tif")

        img2_rotated, rot_mat = ri.rotate_image(img2, 180, return_rot_matrix=True)

        tps, conf = tpd.find_tie_points(img1, img2, mask1=mask, mask2=mask)
        tps_r, conf_r = tpd.find_tie_points(img1, img2_rotated, mask1=mask, mask2=mask_rotated)

        # keep the best tie points
        if tps.shape[0] < tps_r.shape[0]:
            tps = tps_r
            conf = conf_r

            # rotate the tie points
            tps[:, 2:4] = rp.rotate_points(tps[:, 2:4], rot_mat, invert=True)

        if tps.shape[0] < 100:
            print(f"Not enough tie points found between {key} and {img2_name}, skipping.")
            continue

        # merge tps and conf
        merged = np.hstack((tps, conf.reshape(-1, 1)))

        # save the tie points
        output_path = os.path.join(output_fld, f"{key}_{img2_name}_tps.txt")
        np.savetxt(output_path, merged, fmt='%.6f', delimiter=',', header='x1,y1,x2,y2,conf', comments='')

        if display:
            di.display_images([img1, img2], tie_points=tps, tie_points_conf=conf)

        print(key, img2_name, merged.shape)
