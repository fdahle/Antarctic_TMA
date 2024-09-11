"""Estimate the cam id"""

# Library imports
import copy
import pandas as pd
import psycopg2
from typing import Optional, Union

# Local imports
import src.base.connect_to_database as ctd

# Constants
MIN_NR_OF_IMAGES = 3
MAX_STD = 0.005


def estimate_cam_id(image_id: str,
                    use_estimated: bool = False, return_data: bool = False,
                    cam_id_data: Optional[pd.DataFrame] = None,
                    conn: Optional[psycopg2.extensions.connection] = None
                    ) -> Union[None, tuple[None, None], float,
                               tuple[str, Optional[pd.DataFrame]]]:
    """
    Estimates the cam id for a given image based on images with similar properties.
    Args:
        image_id (str): The ID of the image for which the cam id is to be estimated.
        use_estimated (bool): Flag to include estimated cam ids in the analysis.
        return_data (bool): Flag to return the original cam id data.
        cam_id_data (Optional[pd.DataFrame]): Pre-loaded cam id data for images with
            similar properties.
        conn (Optional[Connection]): Database connection object.
    Returns:
        float: A string containing the estimated cam id.
            This return type is provided when `return_data` is False.
        Tuple[float, pd.DataFrame]: A tuple containing the estimated cam id and the
            original subset data `pd.DataFrame` if `return_data` is True.
        None: Returns None if conditions like minimum number of images or maximum standard
            deviation are not met.
    """

    # establish connection to psql if not provided
    if conn is None:
        conn = ctd.establish_connection()

    if cam_id_data is None:
        # get the properties of this image (flight path, etc.)
        sql_string = f"SELECT tma_number, view_direction FROM images WHERE image_id='{image_id}'"
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
        data_ids = [item for sublist in data_ids for item in sublist]  # noqa

        # convert list to a string
        str_data_ids = "('" + "', '".join(data_ids) + "')"

        # get all entries from the same flight and the same viewing direction
        sql_string = f"SELECT image_id, cam_id, cam_id_estimated " \
                     f"FROM images_extracted WHERE image_id IN {str_data_ids}"
        cam_id_data = ctd.execute_sql(sql_string, conn)

    # create a copy for the return
    orig_cam_id_data = copy.deepcopy(cam_id_data)

    # remove the image_id from the image we want to extract information from
    cam_id_data = cam_id_data[cam_id_data['image_id'] != image_id]

    # remove estimated if we don't want to use them
    if use_estimated is False:
        cam_id_data = cam_id_data.loc[cam_id_data['cam_id_estimated'] == False]  # noqa

    # check if we still have data
    if cam_id_data.shape[0] == 0:
        return None

    # count the number of non Nan values
    cam_id_count = cam_id_data.loc[pd.notnull(cam_id_data[f'cam_id'])].shape[0]

    # check if there is a minimum number of images
    if MIN_NR_OF_IMAGES is not None:

        # check if the number of images is below the minimum
        if cam_id_count < MIN_NR_OF_IMAGES:
            return None

    # get most common cam id
    cam_id = cam_id_data['cam_id'].mode().iloc[0]

    return cam_id if not return_data else (cam_id, orig_cam_id_data)
