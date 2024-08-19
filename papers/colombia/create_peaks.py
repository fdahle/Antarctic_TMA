import numpy as np

import src.load.load_image as li
import src.dem.snippets.find_peaks_in_DEM as fpiD

path_dem = "/data/ATM/colombia/data/DEM/DEM_5m.tif"
path_peaks = "/data/ATM/colombia/data/peaks/peaks.shp"


import src.display.display_images as di


def find_peaks_in_dem(dem_path, min_prominence, shapefile_path):

    # load dem
    dem = li.load_image(dem_path)

    # Find peaks in the DEM
    peaks = fpiD.find_peaks_in_dem(dem, no_data_value=-32767)

    # get height of peaks
    heights = dem[peaks[:, 0], peaks[:, 1]]
    print(heights)
    exit()

    # Calculate the prominence of each peak
    heights = dem[peaks_indices[:, 0], peaks_indices[:, 1]]
    prominences = peak_prominences(heights, labeled, wlen=neighborhood_size)[0]

    # Filter peaks by prominence
    filtered_peaks = [(x, y, height, prom) for (x, y), height, prom in zip(peaks_indices, heights, prominences) if
                      prom >= min_prominence]

    # Create a GeoDataFrame to store the peaks
    gdf = gpd.GeoDataFrame(columns=['geometry', 'height', 'prominence'])

    for x, y, height, prom in filtered_peaks:
        gdf = gdf.append({
            'geometry': Point(y, x),  # Note: shapely uses (x, y) as (longitude, latitude)
            'height': height,
            'prominence': prom
        }, ignore_index=True)

    # Set the coordinate reference system (CRS)
    gdf.set_crs(epsg=4326, inplace=True)  # Assuming WGS84, change if necessary

    # Save to shapefile
    gdf.to_file(shapefile_path)

    return gdf


if __name__ == "__main__":
    peaks_gdf = find_peaks_in_dem(path_dem,
                                  min_prominence=15,
                                  shapefile_path=path_peaks)
