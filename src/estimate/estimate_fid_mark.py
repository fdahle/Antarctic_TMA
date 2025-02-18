"""Estimates the fid mark values for a given image"""

# Library imports
import copy
import pandas as pd
import psycopg2
from typing import Optional, Union

# Local imports
import src.base.connect_to_database as ctd

# Constants
MIN_NR_OF_IMAGES = 3
MAX_STD = None


def estimate_fid_mark(image_id: str, key: str,
                      use_estimated: bool = False, return_data: bool = False,
                      fid_mark_data: Optional[pd.DataFrame] = None,
                      conn: Optional[psycopg2.extensions.connection] = None
                      ) -> Union[None, tuple[None, None], tuple[int, int],
                                 tuple[tuple[int, int], Optional[pd.DataFrame]]]:
    """
    Estimates the fid mark values for a given image based on images with similar properties.
    Args:
        image_id (str): The ID of the image for which the fid mark is to be estimated.
        key (str): Key specifying the fid mark (1, 2, 3, 4, 5, 6, 7, 8).
        use_estimated (bool): Flag to include estimated fid marks in the analysis.
        return_data (bool): Flag to return the original fid mark data along with the estimated coordinates.
        fid_mark_data (Optional[pd.DataFrame]): Pre-loaded fid mark data for images with similar properties.
        conn (Optional[Connection]): Database connection object.
    Returns:
        Tuple[int, int]: A tuple containing the estimated x and y coordinates (x_val, y_val).
            This return type is provided when `return_data` is False.
        Tuple[Tuple[int, int], pd.DataFrame]: A tuple containing the estimated coordinates as a tuple of
            integers (x_val, y_val), and the original fid mark data `pd.DataFrame` if `return_data` is True.
        None: Returns None if conditions like minimum number of images or maximum standard deviation are not met.
    """

    # establish connection to psql if not already done
    if conn is None:
        conn = ctd.establish_connection()

    # we only need to get data when it is not provided
    if fid_mark_data is None:
        # get the properties of this image (flight path, etc.)
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
        data_ids = [item for sublist in data_ids for item in sublist]  # noqa

        # convert list to a string
        str_data_ids = "('" + "', '".join(data_ids) + "')"

        # get all entries from the same flight and the same viewing direction
        sql_string = f"SELECT image_id, " \
                     f"fid_mark_1_x, fid_mark_1_y, fid_mark_1_estimated, " \
                     f"fid_mark_2_x, fid_mark_2_y, fid_mark_2_estimated, " \
                     f"fid_mark_3_x, fid_mark_3_y, fid_mark_3_estimated, " \
                     f"fid_mark_4_x, fid_mark_4_y, fid_mark_4_estimated, " \
                     f"fid_mark_5_x, fid_mark_5_y, fid_mark_5_estimated, " \
                     f"fid_mark_6_x, fid_mark_6_y, fid_mark_6_estimated, " \
                     f"fid_mark_7_x, fid_mark_7_y, fid_mark_7_estimated, " \
                     f"fid_mark_8_x, fid_mark_8_y, fid_mark_8_estimated " \
                     f"FROM images_fid_points WHERE image_id IN {str_data_ids}"
        fid_mark_data = ctd.execute_sql(sql_string, conn)

    # create a copy for the return
    orig_fid_mark_data = copy.deepcopy(fid_mark_data)

    # remove the image_id from the image we want to extract information from
    fid_mark_data = fid_mark_data[fid_mark_data['image_id'] != image_id]

    # remove estimated if we don't want to use them
    if use_estimated is False:
        fid_mark_data = fid_mark_data.loc[fid_mark_data[f'fid_mark_{key}_estimated'] == False]  # noqa

    # check if we still have data
    if fid_mark_data.shape[0] == 0:
        return None

    # count the number of non Nan values (x and y should be similar)
    x_count = fid_mark_data.loc[pd.notnull(fid_mark_data[f'fid_mark_{key}_x'])].shape[0]
    y_count = fid_mark_data.loc[pd.notnull(fid_mark_data[f'fid_mark_{key}_y'])].shape[0]

    # check if the counts are similar (should usually always be the case)
    if x_count != y_count:
        return None

    # check if there is a minimum number of images
    if MIN_NR_OF_IMAGES is not None:

        # check if the number of images is below the minimum
        if x_count < MIN_NR_OF_IMAGES or y_count < MIN_NR_OF_IMAGES:
            return None

    # get the std values
    x_std = fid_mark_data[f'fid_mark_{key}_x'].std()
    y_std = fid_mark_data[f'fid_mark_{key}_y'].std()

    # check if there is a maximum standard deviation
    if MAX_STD is not None:

        # check if the standard deviation is above the maximum
        if x_std > MAX_STD or y_std > MAX_STD:
            return None

    # get the mean values
    x_val = fid_mark_data[f'fid_mark_{key}_x'].mean()
    y_val = fid_mark_data[f'fid_mark_{key}_y'].mean()

    # convert to integer
    x_val = int(x_val)
    y_val = int(y_val)

    # save coords as tuple
    coords = (x_val, y_val)

    return (coords, orig_fid_mark_data) if return_data else coords