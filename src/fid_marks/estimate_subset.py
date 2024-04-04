# Package imports
import numpy as np

# Custom imports
import src.base.connect_to_database as ctd

# Constants
MIN_NR_OF_IMAGES = None
MAX_STD = None


def estimate_subsets(image_id, conn=None):
    if conn is None:
        conn = ctd.establish_connection()

    # get the properties of this image (flight path, etc)
    sql_string = f"SELECT tma_number, view_direction, id_cam FROM images WHERE image_id='{image_id}'"
    data_img_props = ctd.execute_sql(sql_string, conn)

    # get the attribute values for this image
    tma_number = data_img_props["tma_number"].iloc[0]
    view_direction = data_img_props["view_direction"].iloc[0]

    # get the images with the same properties
    sql_string = f"SELECT image_id FROM images WHERE tma_number={tma_number} AND " \
                 f"view_direction='{view_direction}'"
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
    sql_string = "SELECT image_id, subset_n_x, subset_n_y, subset_e_x, subset_e_y, " \
                 "subset_s_x, subset_s_y, subset_w_x, subset_w_y, " \
                 "subset_n_estimated, subset_e_estimated, " \
                 "subset_s_estimated, subset_w_estimated " \
                 f"FROM images_properties WHERE image_id IN {str_data_ids}"
    subset_data = ctd.execute_sql(sql_string, conn)

    # init dict to save estimated values
    subset_estimations = {}

    # estimate the values for all directions
    for direction in ["n", "e", "s", "w"]:

        # count the number of non Nan values (x and y should be similar)
        x_count = subset_data.loc[(subset_data[f'subset_{direction}_estimated'] is False) &
                                  (subset_data[f'subset_{direction}_x'] is not np.NaN)].shape[0]

        y_count = subset_data.loc[(subset_data[f'subset_{direction}_estimated'] is False) &
                                  (subset_data[f'subset_{direction}_y'] is not np.NaN)].shape[0]

        if x_count != y_count or x_count < MIN_NR_OF_IMAGES or y_count < MIN_NR_OF_IMAGES:
            continue

        # get the std values
        x_std = subset_data.loc[subset_data[f'subset_{direction}_estimated'] is False,
                                f'subset_{direction}_x'].std()
        y_std = subset_data.loc[subset_data[f'subset_{direction}_estimated'] is False,
                                f'subset_{direction}_y'].std()

        if x_std > MAX_STD or y_std > MAX_STD:
            continue

        # get the average values
        x_val = subset_data.loc[subset_data[f'subset_{direction}_estimated'] is False,
                                f'subset_{direction}_x'].mean()
        y_val = subset_data.loc[subset_data[f'subset_{direction}_estimated'] is False,
                                f'subset_{direction}_y'].mean()

        # convert to integer
        x_val = int(x_val)
        y_val = int(y_val)

        # save in our result dict
        subset_estimations[direction + "_x"] = x_val
        subset_estimations[direction + "_y"] = y_val

    return subset_estimations
