import geopandas as gpd
import shapely

import base.load_shape_data as lsd

poly_min_x = -2668124.52
poly_max_x = -1525426.69
poly_min_y = -1119208.23
poly_max_y = -578854.02


def get_ids_by_coordinates(shape_data, polygon):
    """
    get_ids_by_coordinates(shape_data, polygon)
    Select ids from the TMA shapefile based on a geographic extent. Currently based on EPSG: 3031
    Args:
        shape_data (geopandas): A dataframe containing the data of the shapefile
        polygon (shapely polygon): A shapely polygon from where we want to get the ids
    Returns:
        ids_by_coords (list): A list of ids which the images that are located inside the polygon
    """

    # convert polygon to geopandas
    poly_gpd = gpd.GeoDataFrame(geometry=[polygon], crs=shape_data.crs)  # noqa

    # filter the points that are located inside this polygon
    filtered_shape_data = gpd.sjoin(shape_data, poly_gpd, op="within")

    # get the left-looking images
    images_l = filtered_shape_data['ImageUrl_L']
    images_l = images_l.str.split('/').str[-1].str[:-4]
    images_l = images_l.dropna()
    images_l = images_l.tolist()

    # get the nadir-looking images
    images_v = filtered_shape_data['ImageUrl']
    images_v = images_v.str.split('/').str[-1].str[:-4]
    images_v = images_v.dropna()
    images_v = images_v.tolist()

    # get the right-looking images
    images_r = filtered_shape_data['ImageUrl_R']
    images_r = images_r.str.split('/').str[-1].str[:-4]
    images_r = images_r.dropna()
    images_r = images_r.tolist()

    # merge left, nadir and right ids
    ids_by_coords = images_l + images_v + images_r

    return ids_by_coords


if __name__ == "__main__":

    # create the polygon
    poly = shapely.Polygon([(poly_min_x, poly_min_y), (poly_max_x, poly_min_y),
                            (poly_max_x, poly_max_y), (poly_min_x, poly_max_y)])

    # load the shape_data
    path_shapefile = "/data_1/ATM/data_1/shapefiles/TMA_Photocenters/TMA_pts_20100927.shp"
    shapefile = lsd.load_shape_data(path_shapefile, catch=False)

    # get the ids
    ids = get_ids_by_coordinates(shapefile, poly)

    print(ids)
