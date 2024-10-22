import pandas as pd

import src.base.connect_to_database as ctd

ignore_nan = True

def verify_extracted():

    conn = ctd.establish_connection()

    sql_string = "SELECT * FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # add flight path column
    data['flight_path'] = data['image_id'].str[2:9]

    # get all unique flight paths
    flight_paths = data['flight_path'].unique()

    # check if every flight path has the same focal length
    for flight_path in flight_paths:
        focal_length = data[data['flight_path'] == flight_path]['focal_length'].unique()

        # remove the nan value if we want to ignore it
        if ignore_nan:
            focal_length = focal_length[~pd.isnull(focal_length)]

        if len(focal_length) > 1:
            print(f"Flight path {flight_path} has multiple focal lengths: {focal_length}")

    # check if every flight path has the same cam_id
    for flight_path in flight_paths:
        cam_id = data[data['flight_path'] == flight_path]['cam_id'].unique()

        # remove the nan value if we want to ignore it
        if ignore_nan:
            cam_id = cam_id[~pd.isnull(cam_id)]

        if len(cam_id) > 1:
            print(f"Flight path {flight_path} has multiple cam_ids: {cam_id}")

    # check if every flight path has the same lens cone
    for flight_path in flight_paths:
        lens_cone = data[data['flight_path'] == flight_path]['lens_cone'].unique()

        # remove the nan value if we want to ignore it
        if ignore_nan:
            lens_cone = lens_cone[~pd.isnull(lens_cone)]

        if len(lens_cone) > 1:
            print(f"Flight path {flight_path} has multiple lens cones: {lens_cone}")

if __name__ == "__main__":
    verify_extracted()
