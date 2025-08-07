import os
import numpy as np
import src.sfm_agi2.snippets.create_bundler as cb

project = "casa_grande"
img_type = "aerial_images"
processing_type = "preprocessed"

tp_folder = f"/data/ATM/grenoble_project/{project}/intermediate/{img_type}/tie_points"
img_folder = f"/data/ATM/grenoble_project/{project}/input/{img_type}/{processing_type}_images/images"
bundler_folder = f"/data/ATM/grenoble_project/{project}/intermediate/{img_type}/bundler"

tp_dict = {}
conf_dict = {}

# iterate over all text files in the tie points folder
for tp_file in os.listdir(tp_folder):

    if not tp_file.endswith(".txt"):
        continue

    # get the two image names from the tie point file name
    img1 = tp_file.split("_")[0]
    img2 = tp_file.split("_")[1].replace(".txt", "")

    # create tuple for the image pair
    img_pair = (img1, img2)

    # check if the image pair is already in the dictionary (also reversed)
    if img_pair in tp_dict or (img2, img1) in tp_dict:
        #print(f"Duplicate tie points for {img1} and {img2}, skipping.")
        continue

    # load the tie points
    data = np.loadtxt(os.path.join(tp_folder, tp_file), delimiter=',', skiprows=1)

    # skip if there are too few tie points
    if data.shape[0] < 100:
        continue

    # first 4 columns are the tie points, last column is the confidence
    tps = data[:, :4]
    conf = data[:, 4]

    tp_dict[img_pair] = tps
    conf_dict[img_pair] = conf

cb.create_bundler(img_folder, bundler_folder, tp_dict, conf_dict)