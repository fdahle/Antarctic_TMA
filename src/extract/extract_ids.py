import geopandas as gpd

import base.connect_to_database as ctd

def extract_ids(aoi, image_positions=None,
                ignore_oblique=False, complete_flightpaths=False):

    # if we don't have image positions yet, we can get the from the database
    if image_positions is None:
        # establish connection to psql
        conn = ctd.establish_connection()

        sql_string = "SELECT * FROM images"

    # convert polygon to geopandas
    poly_gpd = gpd.GeoDataFrame(geometry=[polygon], crs=shape_data.crs)  # noqa

    # filter the points that are intersect this polygon
    filtered_shape_data = gpd.sjoin(image_positions, poly_gpd, op="intersect")

    if complete_flightpaths:
        flight_paths = filtered_shape_data['TMA_num'].unique()

        print(flight_paths)


    # get the nadir-looking images
    images_v = filtered_shape_data['ImageUrl']
    images_v = images_v.str.split('/').str[-1].str[:-4]
    images_v = images_v.dropna()
    ids = images_v.tolist()

    if ignore_oblique is False:
        # get the left-looking images
        images_l = filtered_shape_data['ImageUrl_L']
        images_l = images_l.str.split('/').str[-1].str[:-4]
        images_l = images_l.dropna()
        images_l = images_l.tolist()

        # get the right-looking images
        images_r = filtered_shape_data['ImageUrl_R']
        images_r = images_r.str.split('/').str[-1].str[:-4]
        images_r = images_r.dropna()
        images_r = images_r.tolist()

        # merge left, nadir and right ids
        ids = ids + images_l + images_r

    return ids