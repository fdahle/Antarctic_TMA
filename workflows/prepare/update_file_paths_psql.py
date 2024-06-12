# Library imports
import glob
import os
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd

# Constants
BASE_FLD = "/data_1/ATM/data_1"
PATH_DOWNLOADED_FLD = os.path.join(BASE_FLD, "aerial/TMA/downloaded")
PATH_MASK_FLD = os.path.join(BASE_FLD, "aerial/TMA/masked")
PATH_SEGMENTED_FLD = os.path.join(BASE_FLD, "aerial/TMA/segmented/unet")
PATH_XML_FLD = os.path.join(BASE_FLD, "sfm/xml/images")
PATH_RESAMPLED_DOWNLOADED_FLD = os.path.join(BASE_FLD, "aerial/TMA/downloaded_resampled")
PATH_RESAMPLED_MASK_FLD = os.path.join(BASE_FLD, "aerial/TMA/masked_resampled")
# PATH_RESAMPLED_SEGMENTED_FLD = ""

# Variables
overwrite = False

def update_file_paths_psql():

    # establish connection to the database
    conn = ctd.establish_connection()

    sql_string = "SELECT * FROM images_file_paths"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # get image_ids for every folder
    ids_downloaded = [os.path.basename(file) for file in
                      glob.glob(os.path.join(PATH_DOWNLOADED_FLD, '*.tif'))]
    ids_mask = [os.path.basename(file) for file in
                glob.glob(os.path.join(PATH_MASK_FLD, '*.tif'))]
    ids_segmented = [os.path.basename(file) for file in
                     glob.glob(os.path.join(PATH_SEGMENTED_FLD, '*.tif'))]
    ids_xml = [os.path.basename(file) for file in
               glob.glob(os.path.join(PATH_XML_FLD, '*.xml'))]
    ids_resampled_downloaded = [os.path.basename(file) for file in
                                glob.glob(os.path.join(PATH_RESAMPLED_DOWNLOADED_FLD, '*.tif'))]
    ids_resampled_mask = [os.path.basename(file) for file in
                          glob.glob(os.path.join(PATH_RESAMPLED_MASK_FLD, '*.tif'))]
    # ids_resampled_segmented = [os.path.basename(file) for file in
    #                            glob.glob(os.path.join(PATH_RESAMPLED_SEGMENTED_FLD, '*.tif'))]

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Update file paths for {image_id} "
                             f"({updated_entries} already updated)")

        update_dict = {}
        if row["path_downloaded"] is None or overwrite:
            img_pattern = image_id + ".tif"
            downloaded = img_pattern in ids_downloaded
            if downloaded:
                update_dict["path_downloaded"] = os.path.join(PATH_DOWNLOADED_FLD,
                                                              img_pattern)
        if row["path_mask"] is None or overwrite:
            img_pattern = image_id + ".tif"
            mask = img_pattern + ".tif" in ids_mask
            if mask:
                update_dict["path_mask"] = os.path.join(PATH_MASK_FLD,
                                                        img_pattern)
        if row["path_segmented"] is None or overwrite:
            img_pattern = image_id + ".tif"
            segmented = img_pattern in ids_segmented
            if segmented:
                update_dict["path_segmented"] = os.path.join(PATH_SEGMENTED_FLD,
                                                             img_pattern)
        if row["path_xml_file"] is None or overwrite:
            xml_pattern = "MeasuresIm-" + image_id + ".tif.xml"
            xml = xml_pattern in ids_xml
            if xml:
                update_dict["path_xml_file"] = os.path.join(PATH_XML_FLD,
                                                            xml_pattern)
        if row["path_downloaded_resampled"] is None or overwrite:
            img_pattern = image_id + ".tif"
            resampled_downloaded = img_pattern in ids_resampled_downloaded
            if resampled_downloaded:
                update_dict["path_resampled_downloaded"] = os.path.join(PATH_RESAMPLED_DOWNLOADED_FLD,
                                                                        img_pattern)
        if row["path_mask_resampled"] is None or overwrite:
            img_pattern = image_id + ".tif"
            resampled_mask = img_pattern in ids_resampled_mask
            if resampled_mask:
                update_dict["path_mask_resampled"] = os.path.join(PATH_RESAMPLED_MASK_FLD,
                                                                  img_pattern)
        # if row["path_segmented_resampled"] is None or overwrite:
        #     img_pattern = image_id + ".tif"
        #     resampled_segmented = img_pattern in ids_segmented
        #     if resampled_segmented:
        #         update_dict["path_segmented_resampled"] = os.path.join(PATH_RESAMPLED_SEGMENTED_FLD,
        #         img_pattern)

        # skip empty update dicts
        if len(update_dict) == 0:
            continue

        # create sql string from update_dict
        sql_string = "UPDATE images_file_paths SET "
        for key, value in update_dict.items():
            sql_string += f"{key}='{value}', "
        sql_string = sql_string[:-2] + f" WHERE image_id='{image_id}'"

        # update the database
        ctd.execute_sql(sql_string, conn, add_timestamp=False)

        # update the counter
        updated_entries += len(update_dict)


if __name__ == "__main__":
    update_file_paths_psql()