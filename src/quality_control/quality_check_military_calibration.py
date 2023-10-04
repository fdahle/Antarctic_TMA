import pandas as pd

import base.connect_to_db as ctd


def quality_check_military_calibration():
    """

    Returns:

    """
    sql_string = "SELECT image_id, cam_id, focal_length, focal_length_estimated FROM images_extracted"
    data_extracted = ctd.get_data_from_db(sql_string)

    sql_string = "SELECT camera, cfl FROM military_calibration"
    data_military = ctd.get_data_from_db(sql_string)

    # take care of double entries in camera
    # group by camera and aggregate values
    data_military = data_military.groupby('camera').agg({
        'cfl': lambda x: x.mean(skipna=True),
    })

    # reset index to make camera a regular column again
    data_military = data_military.reset_index()

    # merge dataframes on 'cam_id' and 'camera' columns
    data = data_extracted.merge(data_military, left_on='cam_id', right_on='camera', how='inner')

    # drop the redundant 'camera' column
    data = data.drop('camera', axis=1)
    data['difference'] = abs(data['focal_length'] - data['cfl'])

    print(data_extracted)
    print(data_military)
    print(data)


if __name__ == "__main__":
    quality_check_military_calibration()
