import base.load_shape_data as lsd
import os

path_result_points = "/data_1/ATM/data_1/playground/measure_points.shp"
toff_folder = "/data_1/ATM/data_1/playground/georef4/low_res/images"
data = lsd.load_shape_data(path_result_points)

from tqdm import tqdm

import rasterio

coords = [(point.x, point.y) for point in data.geometry]

print(coords)

def find_tiff_for_point(tiff_folder, coordinates):
    """
    Find the TIFF file for each point in the list of coordinates.

    Parameters:
    - tiff_folder: Folder containing geo-referenced TIFF files.
    - coordinates: List of (longitude, latitude) tuples.

    Returns:
    A dictionary where keys are coordinates and values are the names of the TIFF files they are located within.
    """

    tiff_files = [os.path.join(tiff_folder, f) for f in os.listdir(tiff_folder) if f.endswith('.tif')]
    result = {}

    for coord in coordinates:
        for tiff_file in tqdm(tiff_files):
            with rasterio.open(tiff_file) as src:
                if src.bounds.left <= coord[0] <= src.bounds.right and src.bounds.bottom <= coord[1] <= src.bounds.top:
                    result[coord] = tiff_file
                    break

    return result

image_ids = find_tiff_for_point(toff_folder, coords)

print("IDS")
print(image_ids)