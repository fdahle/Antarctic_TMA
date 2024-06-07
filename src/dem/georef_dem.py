import copy
import os.path
from scipy.interpolate import griddata
import numpy as np
from scipy.ndimage import zoom

import src.base.find_tie_points as ftp
import src.display.display_images as di
import src.georef.snippets.convert_image_to_footprint as citf
import src.load.load_rema as lr

import src.dem.minimize_difference as md

buffer_val = 10000

def georef_dem(dem_unref, transform_unref):

    temp_fld = "/home/fdahle/Desktop/tmp"

    # remove outliers
    dem_cleaned = _clean_dem(dem_unref)

    # set no data value to nan
    dem_unref[dem_unref == -99999.0] = np.nan
    dem_cleaned[dem_cleaned == -99999.0] = np.nan

    #di.display_images([dem_unref, dem_cleaned], image_types=['dem', 'dem'])

    # get polygon for the unref_dem
    poly_unref = citf.convert_image_to_footprint(dem_cleaned, transform_unref, no_data_value=-99999.0)

    # make the polygon larger
    poly_large = poly_unref.buffer(buffer_val, join_style='mitre')

    # load georeferenced dem and large dem
    dem_approx= lr.load_rema(poly_unref, zoom_level=32)
    dem_large = lr.load_rema(poly_large, zoom_level=32)

    #di.display_images([dem_unref, dem_approx, dem_large], image_types=['dem', 'dem', 'dem'])

    # upsample the dems with cubic interpolation to fit the approx dem
    zoom_factors = (dem_unref.shape[0] / dem_approx.shape[0], dem_unref.shape[1] / dem_approx.shape[1])
    dem_approx = zoom(dem_approx, zoom_factors, order=3)  # order=3 for cubic interpolation
    dem_large = zoom(dem_large, zoom_factors, order=3)  # order=3 for cubic interpolation

    # find the position of the approx dem in the large dem
    approx_position = (int((dem_large.shape[0] - dem_approx.shape[0]) / 2),
                       int((dem_large.shape[1] - dem_approx.shape[1]) / 2))

    #di.display_images([dem_unref, dem_approx, dem_large], image_types=['dem', 'dem', 'dem'])

    max_order = 4
    step = 50

    step_y = int(step / zoom_factors[0])
    step_x = int(step / zoom_factors[1])

    best_val = float('inf')
    best_tile = (0, 0)
    best_dem = copy.deepcopy(dem_approx)
    best_difference = np.zeros_like(dem_cleaned)

    # here we save all visited tiles
    lst_checked_tiles = []

    for order in range(max_order):

        # create a tile lst
        tiles = []
        for i in range(-order * step_y, (order + 1) * step_y, step_y):
            for j in range(-order * step_x, (order + 1) * step_x, step_x):
                tiles.append((i, j))

        # check all tiles
        for tile in tiles:

            # check if already checked the tile
            if tile in lst_checked_tiles:
                continue
            else:
                lst_checked_tiles.append(tile)

            print("Check Tile: ", tile)

            # get the bounds for that tile
            min_y = approx_position[0] + tile[0]
            max_y = approx_position[0] + tile[0] + dem_approx.shape[0]
            min_x = approx_position[1] + tile[1]
            max_x = approx_position[1] + tile[1] + dem_approx.shape[1]

            # get the tile from the large dem
            dem_tile = copy.deepcopy(dem_large[min_y:max_y, min_x:max_x])

            # set tile to nan where dem_cleaned is nan
            dem_tile[np.isnan(dem_cleaned)] = np.nan

            # get difference between the tile and the unref dem
            difference = np.abs(dem_tile - dem_cleaned)

            # calculate the mean
            mean_val = np.nanmean(difference)

            #save_path = os.path.join(temp_fld, f"diff_{tile[0]}_{tile[1]}.tif")
            #style_config = {"title": f"Shift: {tile[0]}, {tile[1]}, Mean Diff: {mean_val}"}
            #di.display_images([dem_cleaned, dem_tile, difference], image_types=['dem', 'dem', 'rtg'],
            #                  style_config=style_config, save_path=save_path)

            if mean_val < best_val:
                best_val = mean_val
                best_tile = tile
                best_difference = difference
                best_dem = copy.deepcopy(dem_tile)

    print("Best tile: ", best_tile, " with difference: ", best_val)

    #best_difference = best_difference - best_val
    #best_dem = best_dem - best_val

    # minimize the difference


    best_difference[best_difference > 50] = 50

    best_dem, change_val = md.minimize_difference(dem_cleaned, best_dem)

    best_difference = np.abs(best_dem - dem_cleaned)

    print("Adapted dem by: ", change_val)

    di.display_images([dem_cleaned, best_dem, np.abs(best_difference), dem_large],
                      image_types=["dem", "dem", "gtr", "dem"])


def _clean_dem(dem):

    # export dem as tif with cv2
    import cv2
    #cv2.imwrite("/home/fdahle/Desktop/tmp/dem.tif", dem)

    dem_cleaned = copy.deepcopy(dem)
    dem_cleaned[dem_cleaned < 0] = np.nan

    return dem_cleaned

    print("INTERPOLATE")

    # Perform interpolation
    dem_interpolated = np.copy(dem)
    dem_interpolated[invalid_mask] = griddata(
        (x[valid_mask], y[valid_mask]),
        dem[valid_mask],
        (x[invalid_mask], y[invalid_mask]),
        method='cubic'
    )

    return dem_interpolated