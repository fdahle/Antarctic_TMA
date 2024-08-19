# Library imports
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd

# Variables
input_fld = "/data/ATM/data_1/georef/"
overwrite = False


def update_georef_psql():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get the approx positions
    sql_string = "SELECT image_id, st_astext(position_approx) AS position_approx " \
                  "FROM images_extracted"
    footprint_approx = ctd.execute_sql(sql_string, conn)

    # get all the data from shape file
    shp_sat_footprint = gpd.read_file(input_fld + "footprints/sat_footprints.shp")
    shp_sat_center = gpd.read_file(input_fld + "centers/sat_centers.shp")
    shp_sat_oblique = gpd.read_file(input_fld + "footprints/sat_footprints_oblique.shp")
    shp_img_footprint = gpd.read_file(input_fld + "footprints/img_footprints.shp")
    shp_img_center = gpd.read_file(input_fld + "centers/img_centers.shp")
    shp_img_oblique = gpd.read_file(input_fld + "footprints/img_footprints_oblique.shp")
    shp_calc_footprint = gpd.read_file(input_fld + "footprints/calc_footprints.shp")
    shp_calc_center = gpd.read_file(input_fld + "centers/calc_centers.shp")
    shp_calc_oblique = gpd.read_file(input_fld + "footprints/calc_footprints_oblique.shp")

    # get the azimuths as well
    sat_azimuth = pd.read_csv(input_fld + "azimuths/sat_azimuths.csv", header=None)
    img_azimuth = pd.read_csv(input_fld + "azimuths/img_azimuths.csv", header=None)
    calc_azimuth = pd.read_csv(input_fld + "azimuths/calc_azimuths.csv", header=None)
    sat_azimuth.columns = ["image_id", "azimuth"]
    img_azimuth.columns = ["image_id", "azimuth"]
    calc_azimuth.columns = ["image_id", "azimuth"]

    footprints = _merge_gdfs(shp_sat_footprint, shp_img_footprint, shp_calc_footprint)
    centers = _merge_gdfs(shp_sat_center, shp_img_center, shp_calc_center)
    obliques = _merge_gdfs(shp_sat_oblique, shp_img_oblique, shp_calc_oblique)
    azimuths = _merge_pds(sat_azimuth, img_azimuth, calc_azimuth)

    # iterate over all vertical rows (that's the only one in footprints)
    for i, row in tqdm(footprints.iterrows(), total=footprints.shape[0]):

        # get the image id
        image_id = row["image_id"]

        # get the other rows
        row_centers = centers[centers["image_id"] == image_id]
        row_azimuth = azimuths[azimuths["image_id"] == image_id]
        row_approx = footprint_approx[footprint_approx["image_id"] == image_id]

        # extract the data
        footprint = row["geometry"]
        footprint_type = row["footprint_type"]
        position_exact = row_centers["geometry"].values[0]
        position_approx = row_approx["position_approx"].values[0]
        try:
            azimuth = row_azimuth["azimuth"].values[0]
        except (Exception,):
            azimuth = 'NULL'

        # create error vector
        print("TODO: create error vector")
        # print(position_exact, position_approx)

        # load the points
        path_points = f"{input_fld}/images/{footprint_type}/{image_id}_points.txt"
        points = np.loadtxt(path_points)

        # load the points data
        nr_tps = points.shape[0]
        avg_points_quality = np.mean(points[:, 4])
        avg_residuals = np.mean(points[:, 5])

        # update images_extracted
        sql_string = "UPDATE images_extracted SET " \
                     f"footprint_exact=ST_GeomFromText('{footprint}'), " \
                     f"position_exact=ST_GeomFromText('{position_exact}'), " \
                     f"footprint_type='{footprint_type}', " \
                     f"azimuth_exact={azimuth} " \
                     f"WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)

        if image_id == 'CA213731L0077':
            print(sql_string)

        # update images_georef
        sql_string = "UPDATE images_georef SET " \
                        f"footprint_exact=ST_GeomFromText('{footprint}'), " \
                        f"position_exact=ST_GeomFromText('{position_exact}'), " \
                        f"georef_type='{footprint_type}', " \
                        f"azimuth_exact={azimuth}, " \
                        f"nr_tps={nr_tps}, " \
                        f"avg_points_quality={avg_points_quality}, " \
                        f"avg_residuals={avg_residuals} " \
                        f"WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)

        if image_id == 'CA213731L0077':
            print(sql_string)
            exit()
    for i, row in tqdm(obliques.iterrows(), total=obliques.shape[0]):
        # get the image id
        image_id = row["image_id"]

        vertical_id = image_id.replace('31L', '32V')
        vertical_id = vertical_id.replace('33R', '32V')

        # get the center rows
        row_centers = centers[centers["image_id"] == vertical_id]

        # extract the data
        footprint = row["geometry"]
        footprint_type = row["footprint_type"]
        try:
            position_exact = row_centers["geometry"].values[0]
        except (Exception,):
            continue


        # update images_extracted
        sql_string = "UPDATE images_extracted SET " \
                     f"footprint_exact=ST_GeomFromText('{footprint}'), " \
                     f"position_exact=ST_GeomFromText('{position_exact}'), " \
                     f"footprint_type='{footprint_type}' " \
                     f"WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)

def _merge_gdfs(gdf_sat, gdf_img, gdf_calc):

    # merge the footprint files
    gdf_sat['footprint_type'] = 'sat'
    gdf_img['footprint_type'] = 'img'
    gdf_calc['footprint_type'] = 'calc'

    # Concatenate the three GeoDataFrames
    gdf = gpd.GeoDataFrame(pd.concat([gdf_sat, gdf_img, gdf_calc],
                                              ignore_index=True))

    # Sort by image_id and footprint_type to enforce priority
    gdf = gdf.sort_values(by=['image_id', 'footprint_type'], ascending=[True, False])

    # Drop duplicates based on image_id, keeping the first occurrence (which is the highest priority)
    gdf = gdf.drop_duplicates(subset='image_id', keep='first')

    # Reset index (optional)
    gdf = gdf.reset_index(drop=True)

    return gdf

def _merge_pds(pd_sat, pd_img, pd_calc):

    # merge the footprint files
    pd_sat['footprint_type'] = 'sat'
    pd_img['footprint_type'] = 'img'
    pd_calc['footprint_type'] = 'calc'

    # Concatenate the three Pandas
    pdf = pd.concat([pd_sat, pd_img, pd_calc],ignore_index=True)

    # Sort by image_id and footprint_type to enforce priority
    pdf = pdf.sort_values(by=['image_id', 'footprint_type'], ascending=[True, False])

    # Drop duplicates based on image_id, keeping the first occurrence (which is the highest priority)
    pdf = pdf.drop_duplicates(subset='image_id', keep='first')

    # Reset index (optional)
    pdf = pdf.reset_index(drop=True)

    return pdf


if __name__ == "__main__":
    update_georef_psql()
