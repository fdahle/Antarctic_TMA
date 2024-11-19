import numpy as np
from shapely.geometry import box
from rasterio import features

import src.base.resize_image as ri
import src.load.load_rema as lr
import src.load.load_shape_data as lsd

# CONSTANTS
PATH_ROCK_MASK = "/data/ATM/data_1/quantarctica/Quantarctica3/Geology/ADD/ADD_RockOutcrops_Landsat8.shp"


def estimate_dem_quality(dem_abs, modern_dem=None,
                         conf_dem=None,
                         abs_bounds=None,
                         modern_source="REMA32",
                         use_rock_mask=False):

    # load the modern dem if not provided
    if modern_dem is None:

        # historic bounds are required to load the modern dem
        if abs_bounds is None:
            raise ValueError("historic_bounds must be provided if modern_dem is not provided")

        if modern_source == "REMA2":
            zoom_level = 2
        elif modern_source == "REMA10":
            zoom_level = 10
        elif modern_source == "REMA32":
            zoom_level = 32
        else:
            raise ValueError("modern source not supported")

        modern_dem, _ = lr.load_rema(abs_bounds, zoom_level=zoom_level)

    # resize the historic dem to the modern dem
    dem_abs = ri.resize_image(dem_abs, modern_dem.shape)

    # load the rock-mask
    if use_rock_mask:

        # get the rock shapes
        rock_shapes = lsd.load_shape_data(PATH_ROCK_MASK)

        # create polygon from bounds
        min_x, min_y, max_x, max_y = abs_bounds
        bounds_polygon = box(min_x, min_y, max_x, max_y)

        # Filter out rock shapes that are outside the bounds
        rock_shapes = rock_shapes[rock_shapes.geometry.intersects(bounds_polygon)]

        # Calculate the resolution of the DEM
        x_res = (max_x - min_x) / dem_abs.shape[1]
        y_res = (max_y - min_y) / dem_abs.shape[0]

        # Create an affine transformation (mapping pixels to coordinates)
        transform = features.Affine.translation(min_x, max_y) * features.Affine.scale(x_res, -y_res)

        # Rasterize the rock polygons onto the DEM grid
        rock_mask = features.rasterize(
            ((geom, 1) for geom in rock_shapes.geometry),
            out_shape=dem_abs.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
    else:
        rock_mask = None

    quality_dict = {}

    quality_dict = _calc_stats("all", modern_dem, dem_abs, quality_dict)

    if use_rock_mask:
        modern_dem[rock_mask == 0] = np.nan
        quality_dict = _calc_stats("rock", modern_dem, dem_abs, quality_dict)

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
    std_difference = np.nanstd(difference)
    abs_mean_difference = np.nanmean(abs_difference)
    abs_std_difference = np.nanstd(abs_difference)

    quality_dict[f"{calc_type}_mean_difference"] = mean_difference
    quality_dict[f"{calc_type}_std_difference"] = std_difference
    quality_dict[f"{calc_type}_mean_difference_abs"] = abs_mean_difference
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
