import geopandas as gpd
from shapely.wkt import loads

import base.connect_to_db as ctd

def footprints_to_shapes():

    sql_string = "SELECT image_id, ST_AsText(footprint_approx) as footprint_approx FROM images_extracted"
    data = ctd.get_data_from_db(sql_string)
    data = data[data['footprint_approx'].notna()]

    # Convert WKT representations to Shapely geometries
    data['geometry'] = data['footprint_approx'].apply(loads)

    # Create a GeoDataFrame from the dataframe
    gdf = gpd.GeoDataFrame(data)

    # Save the GeoDataFrame as a shapefile
    shapefile_path = '/data_1/ATM/data_1/playground/georeference/all_approx_footprints.shp'
    gdf.to_file(shapefile_path)


if __name__ == "__main__":
    footprints_to_shapes()