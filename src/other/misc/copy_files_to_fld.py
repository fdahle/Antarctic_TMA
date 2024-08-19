import copy
import os.path
import shutil
import src.export.export_tiff as et
from tqdm import tqdm

import src.base.create_mask as cm
import src.base.rotate_image as ri
import src.load.load_image as li

# variables
copy_images = True
copy_masks = True
create_masks = True
overwrite = True

# data variables
add_left = True
add_right = True
fill_gaps = False

# always need to be changed
folder_path = "/home/fdahle/Desktop/agi_test_nature_2"
image_ids = ["CA214732V0031", "CA214732V0032", "CA214732V0033", "CA214732V0034",
             "CA214732V0035", "CA214732V0036", "CA214732V0037", "CA214732V0038",
             "CA214732V0039", "CA214732V0040", "CA214732V0041", "CA214732V0042",
             "CA214732V0043", "CA214732V0044", "CA214732V0045", "CA214732V0046",
             "CA214732V0047", "CA214732V0048", "CA214732V0049", "CA214732V0050",
             "CA214732V0051", "CA214732V0052", "CA214732V0053", "CA214732V0054",
             "CA214732V0055", "CA214732V0056", "CA214732V0057", "CA214732V0058",
             "CA214732V0059", "CA214732V0060", "CA214732V0061", "CA214732V0062",
             "CA214732V0063",
             "CA214832V0064", "CA214832V0065", "CA214832V0066", "CA214832V0067",
             "CA214832V0068", "CA214832V0069", "CA214832V0070", "CA214832V0071",
             "CA214832V0072", "CA214832V0073", "CA214832V0074", "CA214832V0075",
             "CA214832V0076", "CA214832V0077", "CA214832V0078", "CA214932V0161",
             "CA214932V0162", "CA214932V0163", "CA214932V0164", "CA214932V0165",
             "CA214932V0166", "CA214932V0167",
             "CA215032V0244", "CA215032V0245", "CA215032V0246", "CA215032V0247",
             "CA215032V0248", "CA215032V0249", "CA215032V0250", "CA215032V0251",
             "CA215032V0252", "CA215032V0253", "CA215032V0254", "CA215032V0255",
             "CA215132V0289", "CA215132V0290", "CA215132V0291", "CA215132V0292",
             "CA215132V0293", "CA215132V0294", "CA215132V0295", "CA215132V0296",
             "CA215132V0297", "CA215132V0298", "CA215132V0299", "CA215132V0300",
             "CA215132V0301", "CA215132V0302",
             "CA215332V0422", "CA215332V0423", "CA215332V0424", "CA215332V0425",
             "CA215732V0041", "CA215732V0042", "CA215732V0043", "CA215732V0044"
]

"""
image_ids = [
    "CA184832V0165",
    "CA174832V0295",
    "CA216132V0073",
    "CA207532V0302",
    "CA181632V0159",
    "CA183532V0074",
    "CA216332V0157",
    "CA207532V0283",
    "CA184432V0159",
    "CA213432V0285",
    "CA512332V0074",
    "CA207532V0282",
    "CA207432V0218",
    "CA174432V0261",
    "CA184432V0162",
    "CA184732V0025",
    "CA164432V0071",
    "CA212432V0071",
    "CA216332V0164",
    "CA184632V0297",
    "CA184632V0296",
    "CA207532V0307",
    "CA183132V0071",
    "CA184632V0295",
    "CA181332V0161",
    "CA213332V0282",
    "CA174832V0302",
    "CA180132V0162",
    "CA184532V0210",
    "CA207432V0165",
    "CA184532V0219",
    "CA182132V0167",
    "CA164432V0073",
    "CA174132V0282",
    "CA207532V0303",
    "CA182432V0025",
    "CA207532V0305",
    "CA207532V0284",
    "CA184632V0304",
    "CA174232V0071",
    "CA174832V0299",
    "CA135332V0305",
    "CA213932V0298",
    "CA184732V0073",
    "CA173832V0214",
    "CA194232V0288",
    "CA512332V0025",
    "CA203932V0285",
    "CA189332V0298",
    "CA172332V0167",
    "CA204232V0162",
    "CA189332V0300",
    "CA203932V0221",
    "CA203932V0219",
    "CA172132V0288",
    "CA204232V0165",
    "CA189332V0282",
    "CA203932V0283",
    "CA172332V0159",
    "CA203932V0220",
    "CA203932V0212",
    "CA203932V0286",
    "CA203932V0284",
    "CA198232V0025",
    "CA171832V0071",
    "CA512332V0073",
    "CA189332V0291",
    "CA172232V0074",
    "CA172332V0161",
    "CA172332V0154",
    "CA203932V0287",
    "CA203932V0217",
    "CA512432V0031",
    "CA189332V0285",
    "CA203932V0297",
    "CA172332V0157",
    "CA203932V0215",
    "CA189332V0286",
    "CA203932V0213",
    "CA203932V0307",
    "CA203932V0298",
    "CA203932V0214",
    "CA204232V0159",
    "CA204232V0166",
    "CA189332V0288",
    "CA204232V0154",
    "CA216032V0031",
    "CA181332V0083",
    "CA216632V0291",
    "CA181332V0082",
    "CA216632V0294",
    "CA216632V0293",
    "CA216032V0026",
    "CA216632V0304",
    "CA216632V0305",
    "CA216632V0300",
    "CA216632V0286",
    "CA181332V0081",
    "CA216532V0261",
    "CA216632V0308",
    "CA181332V0084",
    "CA216632V0288",
    "CA216632V0295",
    "CA216632V0292",
    "CA216632V0297",
    "CA216632V0307",
    "CA216632V0284",
    "CA216632V0287",
    "CA216632V0298",
    "CA216632V0302",
    "CA216632V0285",
    "CA216632V0290",
    "CA216632V0301",
    "CA216632V0289",
    "CA216632V0296",
    "CA216632V0299",
    "CA216632V0303",
]
"""

# Constants
PATH_IMAGE_FLD = "/data/ATM/data_1/aerial/TMA/downloaded"
PATH_MASK_FLD = "/data/ATM/data_1/aerial/TMA/masks"


def copy_files_to_fld(image_ids, check_for_sky=False):

    # sort image_ids
    image_ids.sort()

    # fill gaps in image_ids
    if fill_gaps:
        import re
        image_dict = {}
        for image_id in image_ids:
            match = re.match(r'CA(\d{4})32V(\d{4})', image_id)
            if match:
                flight_num = match.group(1)
                image_num = int(match.group(2))
                if flight_num not in image_dict:
                    image_dict[flight_num] = []
                image_dict[flight_num].append(image_num)

        # Fill in the gaps for each flight number
        all_filled_ids = []
        for flight_num, image_nums in image_dict.items():
            min_num = min(image_nums)
            max_num = max(image_nums)
            for num in range(min_num, max_num + 1):
                all_filled_ids.append(f'CA{flight_num}32V{num:04}')

        # Return the sorted list of filled image IDs
        image_ids = sorted(all_filled_ids)

    if add_left:
        l_image_ids = []
        for i, elem in enumerate(image_ids):
            l_image_ids.append(elem.replace("32V", "31L"))
    else:
        l_image_ids = []

    if add_right:
        r_image_ids = []
        for i, elem in enumerate(image_ids):
            r_image_ids.append(elem.replace("32V", "33R"))
    else:
        r_image_ids = []

    image_ids = image_ids + l_image_ids + r_image_ids

    # remove duplicate entries
    image_ids = list(set(image_ids))

    import src.base.connect_to_database as ctd
    conn = ctd.establish_connection()

    # get rotation info for the images
    sql_string = "SELECT image_id, sky_is_correct FROM images WHERE image_id IN %s" % str(tuple(image_ids))
    sky_data = ctd.execute_sql(sql_string, conn)

    # create folder if it does not exist
    if os.path.isdir(folder_path) is False:
        os.makedirs(folder_path)

    # check if there's content in the folder
    if len(os.listdir(folder_path)) > 0:
        if overwrite is False:
            raise FileExistsError("Folder is not empty. Set overwrite to True to continue.")
        else:
            # delete all content in the folder
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    shutil.rmtree(file_path)

    if copy_images:
        # create image folder if it does not exist
        image_folder = os.path.join(folder_path, "images")
        if os.path.isdir(image_folder) is False:
            os.makedirs(image_folder)
    else:
        image_folder = None

    if copy_masks:
        # create mask folder if it does not exist
        mask_folder = os.path.join(folder_path, "masks")
        if os.path.isdir(mask_folder) is False:
            os.makedirs(mask_folder)
    else:
        mask_folder = None

    print(image_ids)

    # iterate images
    for image_id in tqdm(image_ids):
        try:
            if copy_images:
                # copy image
                image_path = os.path.join(PATH_IMAGE_FLD, image_id + ".tif")
                image_save_path = os.path.join(image_folder, image_id + ".tif")
                shutil.copy(image_path, image_save_path)
            if copy_masks:

                # create path to mask
                mask_path = os.path.join(PATH_MASK_FLD, image_id + ".tif")
                mask_save_path = os.path.join(mask_folder, image_id + ".tif")

                # check if mask exists
                if os.path.isfile(mask_path) is False:
                    if create_masks:
                        image = li.load_image(image_path)
                        # create mask
                        mask = cm.create_mask(image,
                                              image_id=image_id,
                                              use_default_fiducials=True,
                                              use_database=True,
                                              uint8=True)
                        et.export_tiff(mask, mask_save_path)
                    else:
                        raise FileNotFoundError(f"Mask for image {image_id} not found.")
                else:
                    shutil.copy(mask_path, mask_save_path)

            if check_for_sky:
                sky_correct = sky_data[sky_data['image_id'] == image_id]['sky_is_correct'].iloc[0]

                if sky_correct == False:
                    if copy_images:
                        img = li.load_image(image_save_path)
                        img_r = ri.rotate_image(img, 180)
                        et.export_tiff(img_r, image_save_path, overwrite=True)

                    if copy_masks:
                        mask = li.load_image(mask_save_path)
                        mask_r = ri.rotate_image(mask, 180)
                        et.export_tiff(mask_r, mask_save_path, overwrite=True)

        except Exception as e:
            print(f"Error for image {image_id}: {e}")
            continue

if __name__ == "__main__":
    copy_files_to_fld(image_ids)