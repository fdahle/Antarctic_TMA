# Package imports
import pandas as pd

# Custom imports
import src.base.connect_to_database as ctd

# Constants
MIN_NR_OF_IMAGES = 3
MAX_STD = None

# variables
use_estimated = False


def estimate_focal_length(image_id, conn=None):
    # establish connection to psql if not provided
    if conn is None:
        conn = ctd.establish_connection()

    # get the properties of this image (flight path, etc)
    sql_string = f"SELECT tma_number, view_direction, cam_id FROM images WHERE image_id='{image_id}'"
    data_img_props = ctd.execute_sql(sql_string, conn)

    # get the attribute values for this image
    tma_number = data_img_props["tma_number"].iloc[0]
    view_direction = data_img_props["view_direction"].iloc[0]
    cam_id = data_img_props["cam_id"].iloc[0]

    # get the images with the same properties
    sql_string = f"SELECT image_id FROM images WHERE tma_number={tma_number} AND " \
                 f"view_direction='{view_direction}' AND cam_id={cam_id}"
    data_ids = ctd.execute_sql(sql_string, conn)

    # convert to list and flatten
    data_ids = data_ids.values.tolist()
    data_ids = [item for sublist in data_ids for item in sublist]

    # remove the image_id from the image we want to extract information from
    data_ids.remove(image_id)

    # check if we still have data
    if len(data_ids) == 0:
        return None

    # convert list to a string
    str_data_ids = "('" + "', '".join(data_ids) + "')"

    # get all entries from the same flight and the same viewing direction
    sql_string = f"SELECT image_id, focal_length, focal_length_estimated " \
                 f"FROM images_extracted WHERE image_id IN {str_data_ids}"
    focal_length_data = ctd.execute_sql(sql_string, conn)

    # count the number of non Nan values
    focal_length_data = focal_length_data.loc[(focal_length_data['focal_length_estimated'] == False) &  # noqa
                                              pd.notnull(focal_length_data['focal_length'])]

    # check if there is a minimum number of images
    if MIN_NR_OF_IMAGES is not None:

        # check if the number of images is below the minimum
        if focal_length_data.shape[0] < MIN_NR_OF_IMAGES:
            return None

    # check if the standard deviation is below the maximum
    if MAX_STD is not None:

        # check if the standard deviation is below the maximum
        if focal_length_data['focal_length'].std() > MAX_STD:
            return None

    # get the mean value
    if use_estimated:
        focal_length = focal_length_data['focal_length'].mean()
    else:
        focal_length = focal_length_data.loc[
            focal_length_data['focal_length_estimated'] == False, 'focal_length'].mean()

    return focal_length
