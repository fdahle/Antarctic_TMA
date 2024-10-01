"""Estimate the height"""

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


def estimate_height(image_id: str,
                    use_estimated: bool = False, return_data: bool = False,
                    height_data: Optional[pd.DataFrame] = None,
                    conn: Optional[psycopg2.extensions.connection] = None
                    ) -> Union[None, tuple[None, None], float,
                               tuple[float, Optional[pd.DataFrame]]]:
    """
    Estimates the height for a given image based on images with similar properties.
    Args:
        image_id (str): The ID of the image for which the height is to be estimated.
        use_estimated (bool): Flag to include estimated heights in the analysis.
        return_data (bool): Flag to return the original height data.
        height_data (Optional[pd.DataFrame]): Pre-loaded height data for images with similar properties.
        conn (Optional[Connection]): Database connection object.
    Returns:
        float: A float values containing the estimated height.
            This return type is provided when `return_data` is False.
        Tuple[float, pd.DataFrame]: A tuple containing the estimated height and the
            original subset data `pd.DataFrame` if `return_data` is True.
        None: Returns None if conditions like minimum number of images or maximum standard deviation are not met.
    """

    # establish connection to psql if not already done
    if conn is None:
        conn = ctd.establish_connection()

    # we only need to get data when it is not provided
    if height_data is None:

        # get the properties of this image
        sql_string = f"SELECT tma_number FROM images WHERE image_id='{image_id}'"
        data_img_props = ctd.execute_sql(sql_string, conn)

        # get the attribute values for this image
        tma_number = data_img_props["tma_number"].iloc[0]

        # get the images with the same properties
        sql_string = f"SELECT image_id FROM images WHERE tma_number={tma_number}"
        data_ids = ctd.execute_sql(sql_string, conn)

        # convert to list and flatten
        data_ids = data_ids.values.tolist()
        data_ids = [item for sublist in data_ids for item in sublist]  # noqa

        # convert list to a string
        str_data_ids = "('" + "', '".join(data_ids) + "')"

        # get all entries from the same flight and the same viewing direction
        sql_string = f"SELECT image_id, " \
                     f"height, height_estimated " \
                     f"FROM images_extracted WHERE image_id IN {str_data_ids}"
        height_data = ctd.execute_sql(sql_string, conn)

    # create a copy for the return
    orig_height_data = copy.deepcopy(height_data)

    # remove the image_id from the image we want to extract information from
    height_data = height_data[height_data['image_id'] != image_id]

    # remove estimated if we don't want to use them
    if use_estimated is False:
        height_data = height_data.loc[height_data['height_estimated'] == False]  # noqa

    # check if we still have data
    if height_data.shape[0] == 0:
        return None

    # count the number of non Nan values
    height_count = height_data.loc[pd.notnull(height_data[f'height'])].shape[0]

    # check if there is a minimum number of images
    if MIN_NR_OF_IMAGES is not None:

        # check if the number of images is below the minimum
        if height_count < MIN_NR_OF_IMAGES:
            return None

    # get the std value of height
    std = height_data['height'].std()

    # check if there is a maximum standard deviation
    if MAX_STD is not None:

        # check if the standard deviation is above the maximum
        if std > MAX_STD:
            return None

    # get the mean value
    median_val = height_data['height'].median()

    # round the mean value
    height = int(median_val)

    return (height, orig_height_data) if return_data else height
