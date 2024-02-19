import geopandas as gpd

from shapely.geometry import Polygon

import src.base.connect_to_database as ctd

def extract_ids(aoi, image_positions=None,
                image_directions=['L', 'V', 'R'], complete_flightpaths=False):

<<<<<<< Updated upstream
    # if we don't have image positions yet, we can get them from the database
=======
    # Create points for the corners of the bounding box
    # Order: Bottom Left -> Bottom Right -> Top Right -> Top Left
    points = [(aoi[0], aoi[1]), (aoi[0], aoi[3]), (aoi[0], aoi[3]), (aoi[2], aoi[1]), (aoi[0], aoi[1])]
    # Create a polygon from the points
    poly = Polygon(points)

    # if we don't have image positions yet, we can get the from the database
>>>>>>> Stashed changes
    if image_positions is None:
        # establish connection to psql
        conn = ctd.establish_connection()

        sql_string = "SELECT footprint_approx FROM images"

    # convert polygon to geopandas
    poly_gpd = gpd.GeoDataFrame(geometry=[aoi], crs=3031)  # noqa

    # filter the points that are intersect this polygon
    filtered_shape_data = gpd.sjoin(image_positions, poly_gpd, op="intersect")

    if complete_flightpaths:
        flight_paths = filtered_shape_data['TMA_num'].unique()

        print(flight_paths)
<<<<<<< Updated upstream

    # init the list for ids
    ids = []
=======
    exit()
>>>>>>> Stashed changes

    # get the nadir-looking images
    if "V" in image_directions:
        images_v = filtered_shape_data['ImageUrl']
        images_v = images_v.str.split('/').str[-1].str[:-4]
        images_v = images_v.dropna()
        images_v = images_v.tolist()

        # merge nadir ids
        ids = ids + images_v

    # get the left-looking images
    if "L" in image_directions:
        images_l = filtered_shape_data['ImageUrl_L']
        images_l = images_l.str.split('/').str[-1].str[:-4]
        images_l = images_l.dropna()
        images_l = images_l.tolist()

        # merge left ids
        ids = ids + images_l

    # get the right-looking images
    if "R" in image_directions
        images_r = filtered_shape_data['ImageUrl_R']
        images_r = images_r.str.split('/').str[-1].str[:-4]
        images_r = images_r.dropna()
        images_r = images_r.tolist()

        # merge right ids
        ids = ids + images_r

    return ids
