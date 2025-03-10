# Python imports
import os

# Library imports
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.display.display_shapes as ds
import src.georef.snippets.calc_azimuth as ca
import src.georef.snippets.calc_oblique_footprint as ego

# Variables
georef_types = ["sat", "img", "calc"]
overwrite = True
use_avg_height = True
use_avg_focal_length = True
rema_size=32

# Constants
PATH_GEOREF_FLD = "/data/ATM/data_1/georef/"

flight_paths = None

debug_show_footprints = False

def georef_oblique(georef_type):

    # create connection to psql
    conn = ctd.establish_connection()

    # load the shapefile with center data
    path_center_shp = os.path.join(PATH_GEOREF_FLD, "centers",
                                   georef_type + "_centers.shp")
    center_shp_data = gpd.read_file(path_center_shp)

    # get all vertical image_ids
    image_ids = center_shp_data["image_id"]

    # filter pandas series for flight path
    if flight_paths is not None:
        image_ids = image_ids[image_ids.str[2:6].isin(flight_paths)]

    # convert image_ids to list for sql query
    image_ids = tuple(image_ids)

    if len(image_ids) == 0:
        print("No image_ids found")
        return

    # get height and azimuth for all vertical images
    sql_string = ("SELECT images.image_id, images.altitude, "
                  "ie.height, ie.altimeter_value, ie.focal_length, "
                  "ST_ASTEXT(ie.footprint_exact) AS footprint_exact "
                  "FROM images JOIN images_extracted ie on images.image_id = ie.image_id "
                  f"WHERE images.image_id IN {image_ids}")
    meta_data = ctd.execute_sql(sql_string, conn)

    # combine the height and altimeter value
    meta_data["height"] = meta_data["height"].combine_first(meta_data["altimeter_value"])

    # get average height
    avg_height = np.nanmean(meta_data["height"])

    # load shapefile or create empty gpd with oblique data
    path_oblique_shp = os.path.join(PATH_GEOREF_FLD, "footprints",
                                    georef_type + "_footprints_oblique.shp")
    if os.path.exists(path_oblique_shp) and not overwrite:
        oblique_shp_data = gpd.read_file(path_oblique_shp)
    else:
        oblique_shp_data = gpd.GeoDataFrame(columns=['image_id', 'geometry'], geometry='geometry')
        oblique_shp_data.crs = "EPSG:3031"

    # here the new entries are saved
    new_entries = {'image_id': [], 'geometry': []}
    azimuths = {}

    # iterate over all vertical entries
    for i, center_row in tqdm(center_shp_data.iterrows(), total=center_shp_data.shape[0]):

        # get the image id and center
        image_id = center_row["image_id"]
        center = center_row["geometry"]


        # get the right rows for the current image_id
        meta_row = meta_data[meta_data["image_id"] == center_row["image_id"]]
        if meta_row.empty:
            continue

        # get the correct height
        altitude = meta_row["height"].iloc[0]
        if altitude is None or np.isnan(altitude):
            altitude = meta_row["altimeter_value"].iloc[0]
        if altitude is None or np.isnan(altitude):
            altitude = meta_row["altitude"].iloc[0]
        if altitude is None or np.isnan(altitude) or altitude == -99999:
            if use_avg_height:
                altitude = avg_height
            else:
                print("Could not find altitude for image_id", image_id)
                continue

        vertical_footprint = meta_row[("footprint_exact")]
        vertical_polygon = shapely.from_wkt(vertical_footprint)

        # get the adapted azimuth for this image
        azimuth = ca.calc_azimuth(image_id, center_shp_data)

        if azimuth is None:
            print(f"Could not find azimuth for {image_id}")
            continue

        # save the azimuth
        azimuths[image_id] = azimuth

        # create left and right id
        new_image_ids = [
            image_id.replace('32V', '31L'),
            image_id.replace('32V', '33R')
        ]

        # iterate left and Right
        for new_image_id in new_image_ids:

            if (new_image_id in oblique_shp_data["image_id"].values
                    and overwrite is False):
                print(f"{new_image_id} already exists")
                continue

            if "L" in new_image_id:
                direction = "L"
            else:
                direction = "R"

            sql_string= (f"SELECT focal_length FROM images_extracted "
                         f"WHERE image_id='{new_image_id}'")
            focal_length = ctd.execute_sql(sql_string, conn)

            if focal_length.empty or focal_length["focal_length"].iloc[0] is None:
                if use_avg_focal_length:
                    try:
                        focal_length = np.nanmean(meta_data["focal_length"])
                    except:
                        focal_length = 154.25
                else:
                    print(f"Could not find focal length for {new_image_id}")
                    continue
            else:
                focal_length = focal_length["focal_length"].iloc[0]

            # create oblique polygons
            oblique_polygon = ego.calc_oblique_footprint(center, direction,
                                                          focal_length, altitude,
                                                          azimuth, rema_size)
            if oblique_polygon is None:
                print(f"Could not create oblique polygon for {new_image_id}")
                continue

            style_config = {
                "title": new_image_id,
                "colors": ["blue", "red"]
            }

            if debug_show_footprints:

                ds.display_shapes([vertical_polygon.iloc[0],
                                   oblique_polygon], style_config=style_config)

            # save new entry
            new_entries['image_id'].append(new_image_id)
            new_entries['geometry'].append(oblique_polygon)

    if len(new_entries) == 0:
        print("No new entries")
        return

    # create a GeoDataFrame from new entries
    new_entries_gdf = gpd.GeoDataFrame(new_entries, geometry='geometry')
    new_entries_gdf.crs = oblique_shp_data.crs  # Ensure CRS consistency

    # Combine the old and new data, dropping duplicates (with preference for new entries)
    oblique_shp_data = oblique_shp_data[~oblique_shp_data['image_id'].isin(new_entries_gdf['image_id'])]
    oblique_shp_data = pd.concat([oblique_shp_data, new_entries_gdf], ignore_index=True)

    # save the gpd
    oblique_shp_data.to_file(path_oblique_shp)

    # save the azimuths
    path_azimuths = os.path.join(PATH_GEOREF_FLD, "azimuths",
                                 georef_type + "_azimuths.csv")
    with open(path_azimuths, 'w') as f:
        for key in azimuths.keys():
            f.write("%s,%s\n"%(key, azimuths[key]))

if __name__ == "__main__":
    for georef_type in georef_types:
        georef_oblique(georef_type)