import os
import src.load.load_image as li
import src.export.export_tiff as et
input_path_img = "/data/ATM/data_1/sfm/projects/src_test2/images"
input_path_mask = "/data/ATM/data_1/sfm/projects/src_test2/masks"

for file in os.listdir(input_path_img):
    base_path = os.path.join(input_path_img, file)
    base_name = os.path.basename(file)
    base_name = base_name[:-4]
    mask_path = input_path_mask + "/" + base_name + ".tif"

    img = li.load_image(base_path)
    mask = li.load_image(mask_path)

    # check if the mask is bigger then the image and cut off the mask if necessary
    if img.shape[0] < mask.shape[0]:
        mask = mask[:img.shape[0], :]

        et.export_tiff(mask, mask_path, True)
    # save mask as tif again

    print(img.shape, mask.shape)
