import geopandas as gpd
import glob
import os
import warnings

from shapely import wkt

def modify_shape(file_path, image_id, modus, geometry_data=None, optional_data=None):

    """
    Modify Shapefile to add, delete, or check an image_id.

    Parameters:
    - file_path (str): path to the shapefile
    - image_id (str): the image_id to be added, deleted, or checked
    - modus (str): either 'add', 'delete', or 'check' for the operation
    - geometry_data: WKT string for geometry
    - optional_data: Dictionary containing optional data to be added

    Returns:
    - bool: True if the image_id is present in the shapefile (only for 'check' modus),
            None otherwise.
    """

    if modus == "hard_delete":
        shapefile_files = glob.glob(f'{file_path[:-4]}.*')

        # Delete each file in the list
        for file in shapefile_files:
            os.remove(file)

        return None

    assert image_id is not None

    # Check if the file exists
    if os.path.exists(file_path):
        gdf = gpd.read_file(file_path)
    else:
        gdf = None

    if modus == 'add':

        if type(geometry_data) == str:
            geometry_data = wkt.loads(geometry_data)

        # Create new data row
        new_data = {'image_id': image_id, 'geometry': geometry_data}
        if optional_data:
            new_data.update(optional_data)

        # If GeoDataFrame exists, append, otherwise create new one
        if gdf is None:
            gdf = gpd.GeoDataFrame([new_data], crs="EPSG:3031")
        else:

            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)

                # delete if already existing so that we can append the updated
                if image_id in gdf['image_id'].values:

                    rows_to_delete = gdf[gdf['image_id'] == image_id]
                    gdf = gdf.drop(rows_to_delete.index)

                gdf = gdf.append(new_data, ignore_index=True)

        # Save back to the file
        gdf.to_file(file_path)

    elif modus == 'delete':
        if gdf is None:
            return

        gdf = gdf[gdf.image_id != image_id]

        if len(gdf) > 0:
            gdf.to_file(file_path)
        else:

            shapefile_files = glob.glob(f'{file_path[:-4]}.*')

            # Delete each file in the list
            for file in shapefile_files:
                os.remove(file)

    elif modus == 'check':
        if gdf is None:
            return False
        else:
            return image_id in gdf['image_id'].values

    return None


if __name__ == "__main__":

    base_path = "/data_1/ATM/data_1/playground/georef3"
    test_path = base_path + "/test.shp"

    modify_shape(test_path, "abc", "delete")
    modify_shape(test_path, "12344", "delete")
    modify_shape(test_path, "123", "delete")
    modify_shape(test_path, "2123", "delete")