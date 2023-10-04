import os
import geopandas as gpd
import pandas as pd

from shapely.geometry import Point

id = "CA214732V0030"

shape_path = "/data_1/ATM/data_1/shapefiles/calculated_camera_positions/camera_positions.shp"

def calculate_camera_position(image_id, polygon):

    # calculate the camera position (just by taking the centroid)
    centroid = polygon.centroid
    x, y = centroid.coords[0]

    # Check if shapefile exists, create it if it doesn't
    if not os.path.exists(shape_path):
        crs = {'init': 'epsg:3031'}
        gdf = gpd.GeoDataFrame(columns=['geometry', 'image_id'], crs=crs)
        gdf.to_file(shape_path)

    # Load shapefile
    gdf = gpd.read_file(shape_path)

    # Check if image_id exists in shapefile, update or add point accordingly
    if image_id in gdf['image_id'].values:
        idx = gdf.index[gdf['image_id'] == image_id][0]
        gdf.loc[idx, 'geometry'] = Point(x, y)
    else:
        new_point = gpd.GeoDataFrame({'geometry': [Point(x, y)], 'image_id': [image_id]}, crs=gdf.crs)
        gdf = gpd.GeoDataFrame(pd.concat([gdf, new_point], ignore_index=True), crs=gdf.crs)

    # Save updated shapefile
    gdf.to_file(shape_path, driver='ESRI Shapefile')

if __name__ == "__main__":

    fld="/data_1/ATM/data_1/playground/georeference/"

    import base.load_image_from_file as liff
    img, transform = liff.load_image_from_file(fld + id + ".tif", return_transform=True)

    import image_georeferencing.sub.convert_image_to_footprint as citf
    poly = citf.convert_image_to_footprint(img, id, transform)
    print(poly)

    # Define the properties of the shapefile
    schema = {'geometry': 'Polygon', 'properties': {'image_id': 'int'}}

    import fiona
    from shapely.geometry import mapping

    with fiona.open(fld + id + "_footprint.shp", 'w', 'ESRI Shapefile', schema) as file:
        file.write({'geometry': mapping(poly), 'properties': {'image_id': 1}})

    calculate_camera_position(id, poly)