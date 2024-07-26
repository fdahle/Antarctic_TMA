import geopandas as gpd
import src.base.connect_to_database as ctd

# Variables
input_fld = "/data/ATM/data_1/georef/"
methods = ["sat"]

def update_georef_psql_oblique(method):

    # get path to shapefile
    path_shp_file = input_fld + "/" + method + "_oblique.shp"

    # open shapefile with the footprints
    footprints = gpd.read_file(path_shp_file)
    print(footprints)

    conn = ctd.establish_connection()

    # iterate over the rows
    for index, row in footprints.iterrows():
        image_id = row['image_id']
        footprint_exact = row['geometry']

        # create sql string to update
        sql_string = (f"UPDATE images_extracted SET "
                      f"footprint_exact = ST_GeomFromText('{footprint_exact}', 3031) "
                      f"WHERE image_id = '{image_id}'")
        ctd.execute_sql(sql_string, conn)

if __name__ == "__main__":

    for _method in methods:
        update_georef_psql_oblique(_method)