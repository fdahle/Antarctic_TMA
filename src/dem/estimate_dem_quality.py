import numpy as np
from shapely.geometry import box
from rasterio import features

import src.base.resize_image as ri
import src.load.load_rema as lr
import src.load.load_shape_data as lsd

# CONSTANTS
PATH_ROCK_MASK = "/data/ATM/data_1/quantarctica/Quantarctica3/Geology/ADD/ADD_RockOutcrops_Landsat8.shp"


def estimate_dem_quality(dem_abs, modern_dem=None,
                         mask=None):

    # init quality dict
    quality_dict = {}

    # get the quality of the whole dem
    quality_dict = _calc_stats("all", modern_dem, dem_abs, quality_dict)

    # apply the mask
    if mask is not None:

        modern_dem[mask == 0] = np.nan
        quality_dict = _calc_stats("mask", modern_dem, dem_abs, quality_dict)

    # Convert all numpy float32 values in quality_dict to standard Python float
    quality_dict = {key: float(value) if isinstance(value, np.float32) else value for key, value in
                    quality_dict.items()}

    return quality_dict


def _calc_stats(calc_type, modern_dem, historic_dem, quality_dict):
    # get a difference map
    difference = modern_dem - historic_dem
    abs_difference = np.abs(difference)

    # calculate the mean difference
    mean_difference = np.nanmean(difference)
    median_difference = np.nanmedian(difference)
    std_difference = np.nanstd(difference)
    abs_mean_difference = np.nanmean(abs_difference)
    abs_median_difference = np.nanmedian(abs_difference)
    abs_std_difference = np.nanstd(abs_difference)

    quality_dict[f"{calc_type}_mean_difference"] = mean_difference
    quality_dict[f"{calc_type}_median_difference"] = median_difference
    quality_dict[f"{calc_type}_std_difference"] = std_difference
    quality_dict[f"{calc_type}_mean_difference_abs"] = abs_mean_difference
    quality_dict[f"{calc_type}_median_difference_abs"] = abs_median_difference
    quality_dict[f"{calc_type}_difference_abs_std"] = abs_std_difference

    rmse = np.sqrt(np.nanmean((modern_dem - historic_dem) ** 2))
    mae = np.nanmean(np.abs(modern_dem - historic_dem))
    mad = np.nanmedian(np.abs(modern_dem - historic_dem - np.nanmedian(modern_dem - historic_dem)))

    quality_dict[f"{calc_type}_rmse"] = rmse
    quality_dict[f"{calc_type}_mae"] = mae
    quality_dict[f"{calc_type}_mad"] = mad

    # Flatten the DEM arrays
    modern_dem_flat = modern_dem.flatten()
    historic_dem_flat = historic_dem.flatten()

    # Create a mask for valid (non-NaN) values in both DEMs
    valid_mask = ~np.isnan(modern_dem_flat) & ~np.isnan(historic_dem_flat)

    # Apply the mask to filter out NaNs
    modern_dem_valid = modern_dem_flat[valid_mask]
    historic_dem_valid = historic_dem_flat[valid_mask]

    # Calculate the correlation coefficient on the valid data
    correlation = np.corrcoef(modern_dem_valid, historic_dem_valid)[0, 1]
    quality_dict[f"{calc_type}_correlation"] = correlation

    return quality_dict
