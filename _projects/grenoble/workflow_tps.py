# define path to project
input_fld = "/data/ATM/grenoble_project/data/iceland/input/"
work_fld = "/data/ATM/grenoble_project/data/iceland/intermediate/"

# we use the preprocessed aerial images
input_fld = input_fld + "aerial_images/"
input_img_fld = input_fld + "preprocessed_images/"

import src.load.load_image as li
import src.display.display_images as di

# load the mask
mask = li.load_image(input_img_fld + "image_mask.tif")
mask=1-mask

# get all tif tiles in the input folder starting with "F_"
import glob

# get list of image paths and image names
lst_image_pth = glob.glob(input_img_fld + "F-*.tif")
lst_image_pth.sort()
lst_image_nms = [p.split("/")[-1][:-4] for p in lst_image_pth]

print(lst_image_nms)

import src.base.find_tie_points as ftp

# load the footprints
import src.load.load_shape_data as lsd

footprints = lsd.load_shape_data(input_fld + "image_footprints.shp")

print(footprints.columns)
geom_map = {row['Image ID'].replace('.tif', ''): row['geometry'] for _, row in footprints.iterrows()}
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

output_fld = "/data/ATM/grenoble_project/data/iceland/intermediate/tie_points/aerial/"

for key, value in overlapping_images.items():

    img1 = li.load_image(input_img_fld + key + ".tif")

    print(key, value)

    for img2_name in value:

        print("Processing tie points for:", key, img2_name)

        img2 = li.load_image(input_img_fld + img2_name + ".tif")

        tps, conf = tpd.find_tie_points(img1, img2, mask1=mask, mask2=mask)

        # merge tps and conf
        merged = np.hstack((tps, conf.reshape(-1, 1)))

        # save the tie points
        output_path = os.path.join(output_fld, f"{key}_{img2_name}_tps.txt")
        np.savetxt(output_path, merged, fmt='%.6f', delimiter=',', header='x1,y1,x2,y2,conf', comments='')

        print(key, img2_name, merged.shape)
