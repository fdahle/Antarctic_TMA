import geopandas as gpd
from shapely import wkt

import src.base.connect_to_database as ctd

save_path = "/data/ATM/data_1/georef/centers/centers.shp"

def save_camera_centers():

    conn = ctd.establish_connection()

    sql_string = "SELECT image_id, ST_AsText(position_exact) AS position_exact " \
                 "FROM images_georef_3 WHERE position_exact IS NOT NULL"
    data = ctd.execute_sql(sql_string, conn)

    # Convert the fetched data to a GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=data['position_exact'].apply(wkt.loads),
                           crs="EPSG:3031")
    gdf.to_file(save_path, driver='ESRI Shapefile')

if __name__ == "__main__":
    save_camera_centers()
