# Package imports
import pandas as pd

# Custom imports
import src.base.connect_to_database as ctd

# Constants
MIN_NR_OF_IMAGES = 3
MAX_STD = None


def estimate_subset(image_id, key, conn=None):
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
    sql_string = f"SELECT image_id, subset_{key}_x, subset_{key}_y, subset_{key}_estimated " \
                 f"FROM images_fid_points WHERE image_id IN {str_data_ids}"
    subset_data = ctd.execute_sql(sql_string, conn)

    # count the number of non Nan values (x and y should be similar)
    x_count = subset_data.loc[(subset_data[f'subset_{key}_estimated'] == False) &  # noqa
                              pd.notnull(subset_data[f'subset_{key}_x'])].shape[0]
    y_count = subset_data.loc[(subset_data[f'subset_{key}_estimated'] == False) &  # noqa
                              pd.notnull(subset_data[f'subset_{key}_y'])].shape[0]

    # check if the counts are similar (should usually never happen)
    if x_count != y_count:
        return None

    # check if there is a minimum number of images
    if MIN_NR_OF_IMAGES is not None:

        # check if the number of images is below the minimum
        if x_count < MIN_NR_OF_IMAGES or y_count < MIN_NR_OF_IMAGES:
            return None

    # get the std values
    x_std = subset_data.loc[subset_data[f'subset_{key}_estimated'] == False,  # noqa
                            f'subset_{key}_x'].std()
    y_std = subset_data.loc[subset_data[f'subset_{key}_estimated'] == False,  # noqa
                            f'subset_{key}_y'].std()

    # check if there is a maximum standard deviation
    if MAX_STD is not None:

        # check if the standard deviation is above the maximum
        if x_std > MAX_STD or y_std > MAX_STD:
            return None

    # get the mean values
    x_val = subset_data.loc[subset_data[f'subset_{key}_estimated'] == False,  # noqa
                            f'subset_{key}_x'].mean()
    y_val = subset_data.loc[subset_data[f'subset_{key}_estimated'] == False,  # noqa
                            f'subset_{key}_y'].mean()

    # convert to integer
    x_val = int(x_val)
    y_val = int(y_val)

    return x_val, y_val
