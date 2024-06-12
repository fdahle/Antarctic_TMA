"""Estimates the subset values for a given image"""

# Library imports
import copy
import pandas as pd
import psycopg2
from psycopg2 import extensions
from typing import Optional, Union

# Local imports
import src.base.connect_to_database as ctd

# Constants
MIN_NR_OF_IMAGES = 3
MAX_STD = None


def estimate_subset(image_id: str,
                    key: str,
                    use_estimated: bool = False,
                    return_data: bool = False,
                    subset_data: Optional[pd.DataFrame] = None,
                    conn: Optional[psycopg2.extensions.connection] = None
                    ) -> Union[None, tuple[None, None], tuple[int, int],
                               tuple[tuple[int, int], Optional[pd.DataFrame]]]:
    """
    Estimates the subset values for a given image based on images with similar properties.
    Args:
        image_id (str): The ID of the image for which the subset is to be estimated.
        key (str): Key specifying the subset direction ('n', 'e', 's', 'w').
        use_estimated (bool): Flag to include estimated subsets in the analysis.
        return_data (bool): Flag to return the original subset data along with the estimated coordinates.
        subset_data (Optional[pd.DataFrame]): Pre-loaded subset data for images with similar properties.
        conn (Optional[Connection]): Database connection object.
    Returns:
        Tuple[int, int]: A tuple containing the estimated x and y coordinates (x_val, y_val).
            This return type is provided when `return_data` is False.
        Tuple[Tuple[int, int], pd.DataFrame]: A tuple containing the estimated coordinates as a tuple of
            integers (x_val, y_val), and the original subset data `pd.DataFrame` if `return_data` is True.
        None: Returns None if conditions like minimum number of images or maximum standard deviation are not met.
    """

    # establish connection to psql if not already done
    if conn is None:
        conn = ctd.establish_connection()

    # we only need to get data when it is not provided
    if subset_data is None:
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
                     f"subset_n_x, subset_n_y, subset_n_estimated, " \
                     f"subset_e_x, subset_e_y, subset_e_estimated, " \
                     f"subset_s_x, subset_s_y, subset_s_estimated, " \
                     f"subset_w_x, subset_w_y, subset_w_estimated " \
                     f"FROM images_fid_points WHERE image_id IN {str_data_ids}"
        subset_data = ctd.execute_sql(sql_string, conn)

    # create a copy for the return
    orig_subset_data = copy.deepcopy(subset_data)

    # remove the image_id from the image we want to extract information from
    subset_data = subset_data[subset_data['image_id'] != image_id]

    # remove estimated if we don't want to use them
    if use_estimated is False:
        subset_data = subset_data.loc[subset_data[f'subset_{key}_estimated'] == False]  # noqa

    # check if we still have data
    if subset_data.shape[0] == 0:
        return None

    # count the number of non Nan values (x and y should be similar)
    x_count = subset_data.loc[pd.notnull(subset_data[f'subset_{key}_x'])].shape[0]
    y_count = subset_data.loc[pd.notnull(subset_data[f'subset_{key}_y'])].shape[0]

    # check if the counts are similar (should usually always be the case)
    if x_count != y_count:
        return None, None if return_data else None

    # check if there is a minimum number of images
    if MIN_NR_OF_IMAGES is not None:

        # check if the number of images is below the minimum
        if x_count < MIN_NR_OF_IMAGES or y_count < MIN_NR_OF_IMAGES:
            return None, None if return_data else None

    # get the std values
    x_std = subset_data[f'subset_{key}_x'].std()
    y_std = subset_data[f'subset_{key}_y'].std()

    # check if there is a maximum standard deviation
    if MAX_STD is not None:

        # check if the standard deviation is above the maximum
        if x_std > MAX_STD or y_std > MAX_STD:
            return None, None if return_data else None

    # get the mean values
    x_val = subset_data[f'subset_{key}_x'].mean()
    y_val = subset_data[f'subset_{key}_y'].mean()

    # convert to integer
    x_val = int(x_val)
    y_val = int(y_val)

    # save coords as tuple
    coords = (x_val, y_val)

    return (coords, orig_subset_data) if return_data else coords
