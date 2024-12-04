# Python imports
import os

# Library imports
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# Local imports
import src.georef.snippets.calc_camera_position as ccp

# Variables
georef_types = ["calc"]
overwrite = False

# Constants
PATH_GEOREF_FLD = "/data/ATM/data_1/georef/"


def georef_center(georef_type):

    # load the shapefile with footprint data
    path_footprint_shp = os.path.join(PATH_GEOREF_FLD, "footprints",
                                     georef_type + "_footprints.shp")
    footprint_shp_data = gpd.read_file(path_footprint_shp)

    # load shapefile or create empty gpd with center data
    path_center_shp = os.path.join(PATH_GEOREF_FLD, "centers",
                                   georef_type + "_centers.shp")
    if os.path.exists(path_center_shp) and not overwrite:
        center_shp_data = gpd.read_file(path_center_shp)
    else:
        center_shp_data = gpd.GeoDataFrame(columns=['image_id', 'geometry'], geometry='geometry')
        center_shp_data.crs = "EPSG:3031"

    # here the new entries are saved
    new_entries = {'image_id': [], 'geometry': []}

    # iterate over all rows
    for i, shp_row in tqdm(footprint_shp_data.iterrows(), total=footprint_shp_data.shape[0]):

        # get the image id
        image_id = shp_row["image_id"]

        # check if we need to overwrite
        if (image_id in center_shp_data["image_id"].values
                and overwrite is False):
            print(f"{image_id} already exists")
            continue

        # calculate center
        center = ccp.calc_camera_position(image_id, shp_row["geometry"])

        # save new entry
        new_entries['image_id'].append(image_id)
        new_entries['geometry'].append(center)

    if len(new_entries) == 0:
        print("No new entries")
        return

    # create a GeoDataFrame from new entries
    new_entries_gdf = gpd.GeoDataFrame(new_entries, geometry='geometry')
    new_entries_gdf.crs = center_shp_data.crs  # Ensure CRS consistency

    # Combine the old and new data, dropping duplicates (with preference for new entries)
    center_shp_data = center_shp_data[~center_shp_data['image_id'].isin(new_entries_gdf['image_id'])]
    center_shp_data = pd.concat([center_shp_data, new_entries_gdf], ignore_index=True)

    # save the gpd
    center_shp_data.to_file(path_center_shp)


if __name__ == "__main__":

    for g_type in georef_types:
        georef_center(g_type)