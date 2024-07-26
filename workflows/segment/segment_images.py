# Library imports
import numpy as np
import os.path
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.base.resize_image as ri
import src.export.export_tiff as et
import src.load.load_image as li
import src.segment.segment_image as si
import src.segment.improve_segmentation as ise

# Constants
OVERWRITE = False
BORDER_WIDTH = 700
STARTING_SIZE = 4000
PATH_SEGMENTATION_FLD = "/data/ATM/data_1/aerial/TMA/segmented/unet/"

def segment_images(improve_images=False):

    # establish connection to psql
    conn = ctd.establish_connection()

    # get a list of all images from the database
    sql_string = "SELECT image_id FROM images"
    data = ctd.execute_sql(sql_string, conn)
    image_ids = data["image_id"].tolist()

    # iterate all images
    for image_id in tqdm(image_ids):

        path_segmented_img = PATH_SEGMENTATION_FLD + image_id + ".tif"

        # check if we already segmented the image
        if os.path.isfile(path_segmented_img) and not OVERWRITE:
            print("Image already segmented")
            continue

        # load the image
        try:
            image = li.load_image(image_id)
        except (Exception,):
            print(f"Image {image_id} could not be loaded")
            continue

        # remove the mask from the image
        image_small = image[BORDER_WIDTH:-BORDER_WIDTH, BORDER_WIDTH:-BORDER_WIDTH]

        # save the shape of the small image
        small_shape = image_small.shape

        # already resize to a smaller size to speed up the segmentation
        try:
            image_small = ri.resize_image(image_small,
                                          (STARTING_SIZE, STARTING_SIZE))
        except (Exception,):
            print(f"Image {image_id} could not be resized")
            continue

        # status flag for successful segmentation
        success = False

        # segment the image
        while True:
            try:

                # segment the image
                segmented, probabilities, model_name = si.segment_image(image_small,
                                                                        min_threshold=0.9,
                                                                        return_model_name=True)

                # set the status flag and escape the loop
                success = True
                break

            except RuntimeError as e:

                # no memory -> make image smaller
                if 'out of memory' in str(e):

                    # decrease image_shape by 10 %
                    smaller_shape = (int(image_small.shape[0] * 0.9), int(image_small.shape[1] * 0.9))

                    # resize image
                    image_small = ri.resize_image(image_small, smaller_shape)

                    continue

                if 'Sizes of tensors must match' in str(e):

                    # Resize image to nearest power of 2 plus 1
                    h, w = image_small.shape
                    new_h = (2 ** (h - 1).bit_length() // 2) - 1
                    new_w = (2 ** (w - 1).bit_length() // 2) - 1

                    # resize image
                    image_small = ri.resize_image(image_small, (new_w, new_h))

                    continue

                # some other error
                else:
                    print(f"Runtime error: {e}")
                    break

        # skip if the segmentation was not successful
        if not success:
            print("Segmentation failed")
            continue

        # improve the segmentation
        if improve_images:
            try:
                segmented_improved = ise.improve_segmentation(image_id, segmented, probabilities)
            except (Exception,):
                print(f"Image {image_id} could not be improved")
                continue
        else:
            segmented_improved = segmented

        # resize the segmented image back to the original size
        try:
            segmented_final = ri.resize_image(segmented_improved,
                                              small_shape, interpolation='nearest')
        except (Exception,):
            print(f"Image {image_id} could not be resized back")
            continue

        # add borders back to the segmented
        segmented_final = np.pad(segmented_final, ((BORDER_WIDTH, BORDER_WIDTH), (BORDER_WIDTH, BORDER_WIDTH)),
                                 mode='constant', constant_values=7)

        # check to see if the segmented size is correct
        if segmented_final.shape != image.shape:
            print(f"Segmented image shape is wrong: {segmented_final.shape} vs {image.shape}")
            continue

        # get the number of unique values per class of the segmented images
        total_number_of_pixels = segmented.shape[0] * segmented.shape[1]
        uniq, counts = np.unique(segmented, return_counts=True)

        # get the percentages from the image & fill the update_dict
        update_dict = {}
        labels = ["perc_ice", "perc_snow", "perc_rocks", "perc_water",
                  "perc_clouds", "perc_sky", "perc_other"]

        try:
            # iterate all percentages
            for i in range(7):

                class_val = i + 1

                # get position of value in uniq
                if class_val in uniq:
                    class_idx = np.where(uniq == class_val)[0][0]

                    class_count = counts[class_idx]
                    update_dict[labels[i]] = round(class_count / total_number_of_pixels * 100, 2)
                else:
                    update_dict[labels[i]] = 0

            # create sql_string from update_dict
            sql_string = "UPDATE images_segmentation SET "
            for key in update_dict:
                sql_string += f"{key}={update_dict[key]}, "
            sql_string = sql_string + "labelled_by='unet', "
            sql_string = sql_string + f"model_name='{model_name}', "
            sql_string = sql_string[:-2] + f" WHERE image_id='{image_id}';"

            ctd.execute_sql(sql_string, conn)

            # save the segmented image
            et.export_tiff(segmented_final, path_segmented_img, use_lzw=True)

            # update the path in database
            sql_string = (f"UPDATE images_file_paths SET path_segmented='{path_segmented_img}' "
                          f"WHERE image_id='{image_id}';")
            ctd.execute_sql(sql_string, conn, add_timestamp=False)
        except (Exception,):
            print(f"Image {image_id} could not be saved")
            continue


if __name__ == "__main__":
    segment_images()