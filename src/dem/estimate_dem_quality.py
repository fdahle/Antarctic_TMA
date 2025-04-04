import numpy as np
import xdem
import src.display.display_images as di

debug_plot = False

def estimate_dem_quality(dem, modern_dem,
                         mask=None, max_slope=None,
                         slope_transform=None):

    # copy to not change original
    dem = dem.copy()

    if dem.shape != modern_dem.shape:
        raise ValueError("The DEMs must have the same shape.")

    if mask is not None:
        if mask.shape != dem.shape:
            raise ValueError("The mask must have the same shape as the DEMs.")
    else:
        mask = np.ones(dem.shape, dtype=bool)

    mask[np.isnan(dem)] = 0
    mask[np.isnan(modern_dem)] = 0

    if max_slope is not None:
        if slope_transform is None:
            raise ValueError("Transform must be provided if max_slope is specified.")

    if max_slope is not None:

        dem_arr = xdem.DEM.from_array(dem, slope_transform, crs=3031)

        # get slope
        slope = xdem.terrain.slope(dem_arr)
        slope_mask = slope < max_slope
        slope_mask = slope_mask.data
        slope_mask = np.ma.getdata(slope_mask)


        mask[slope_mask == 0] = 0

    if debug_plot:
        if max_slope is None:
            di.display_images([dem, modern_dem, mask] )
        else:
            di.display_images([dem, modern_dem, slope.data, mask] )

    # print percentage of masked pixels
    print(f"  Percentage of masked pixels: { 100 - (np.count_nonzero(mask) / mask.size * 100):.2f}%")

    # apply the mask to the DEMs
    dem[mask == 0] = np.nan

    # get the difference between the two DEMs
    difference = modern_dem - dem
    abs_difference = np.abs(difference)

    quality_dict = {
        "mean_difference": np.nanmean(difference),
        "median_difference": np.nanmedian(difference),
        "std_difference": np.nanstd(difference),
        "mean_difference_abs": np.nanmean(abs_difference),
        "median_difference_abs": np.nanmedian(abs_difference),
        "difference_abs_std": np.nanstd(abs_difference),
        "rmse": np.sqrt(np.nanmean(difference**2)),
        "mae": np.nanmean(abs_difference),
        "mad": np.nanmedian(np.abs(difference - np.nanmedian(difference)))
    }

    # Correlation (only valid values)
    flat_modern = modern_dem.flatten()
    flat_historic = dem.flatten()
    valid = ~np.isnan(flat_modern) & ~np.isnan(flat_historic)

    if np.count_nonzero(valid) > 1:
        quality_dict["correlation"] = np.corrcoef(flat_modern[valid], flat_historic[valid])[0, 1]
    else:
        quality_dict["correlation"] = np.nan

    return quality_dict