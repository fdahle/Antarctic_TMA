project_name = "crane_glacier"

import os
import glob
import numpy as np

project_path = f"/data/ATM/data_1/sfm/agi_projects/{project_name}"
output_fld = os.path.join(project_path, "output")

# get path for the dem of our project
pattern = os.path.join(output_fld, f"{project_name}_dem_absolute_*.tif")
lst_dem_c = glob.glob(pattern)
if len(lst_dem_c) == 0:
    print("No corrected dem found")
    exit()
path_corrected_dem = lst_dem_c[0]

# load our dem
import src.load.load_image as li
dem_corrected, transform_corrected = li.load_image(path_corrected_dem, return_transform=True)
dem_corrected[dem_corrected == -9999] = np.nan  # set nodata to nan

# load dem of ryan
glac_name = project_name.split("_")[0].capitalize()
pth_ryan_dem = f"/data/ATM/data_1/ryan_data/hist_dems/North_and_Barrows_2024_DEM_{glac_name}_1968.tif"
dem_ryan, transform_ryan = li.load_image(pth_ryan_dem, return_transform=True)
dem_ryan[dem_ryan == -32767] = np.nan  # set nodata to nan

# get bounds of both dems
import src.base.calc_bounds as cb
bounds_corrected = cb.calc_bounds(transform_corrected, dem_corrected.shape)
bounds_ryan = cb.calc_bounds(transform_ryan, dem_ryan.shape)

# Find overlap
overlap_left = max(bounds_corrected[0], bounds_ryan[0])
overlap_right = min(bounds_corrected[2], bounds_ryan[2])
overlap_bottom = max(bounds_corrected[1], bounds_ryan[1])
overlap_top = min(bounds_corrected[3], bounds_ryan[3])
if overlap_right <= overlap_left or overlap_top <= overlap_bottom:
    raise ValueError("DEMs do not overlap!")

# Crop both DEMs to the overlapping extent using their native grid
import rasterio
from rasterio.windows import from_bounds
window_corrected = from_bounds(overlap_left, overlap_bottom, overlap_right, overlap_top, transform_corrected)
dem_corrected_crop = dem_corrected[
    int(window_corrected.row_off):int(window_corrected.row_off + window_corrected.height),
    int(window_corrected.col_off):int(window_corrected.col_off + window_corrected.width)
]
transform_corrected_crop = rasterio.windows.transform(window_corrected, transform_corrected)

window_ryan = from_bounds(overlap_left, overlap_bottom, overlap_right, overlap_top, transform_ryan)
dem_ryan_crop = dem_ryan[
    int(window_ryan.row_off):int(window_ryan.row_off + window_ryan.height),
    int(window_ryan.col_off):int(window_ryan.col_off + window_ryan.width)
]
transform_ryan_crop = rasterio.windows.transform(window_ryan, transform_ryan)



# Reproject Ryan DEM crop to corrected DEM crop grid
dst_shape = dem_corrected_crop.shape
dem_ryan_resampled = np.empty(dst_shape, dtype=np.float32)

from rasterio.warp import reproject, Resampling
reproject(
    source=dem_ryan_crop,
    destination=dem_ryan_resampled,
    src_transform=transform_ryan_crop,
    src_crs="EPSG:3031",  # adjust if needed
    dst_transform=transform_corrected_crop,
    dst_crs="EPSG:3031",
    resampling=Resampling.bilinear
)


# Create valid mask and calculate difference
mask = (~np.isnan(dem_corrected_crop)) & (~np.isnan(dem_ryan_resampled))

# Calculate absolute difference
difference = dem_ryan_resampled - dem_corrected_crop
abs_difference = np.abs(difference)
#difference[mask] = np.nan

# get quality
quality_dict = {
    "mean_difference": np.nanmean(difference),
    "median_difference": np.nanmedian(difference),
    "std_difference": np.nanstd(difference),
    "mean_difference_abs": np.nanmean(abs_difference),
    "median_difference_abs": np.nanmedian(abs_difference),
    "difference_abs_std": np.nanstd(abs_difference),
    "rmse": np.sqrt(np.nanmean(difference ** 2)),
    "mae": np.nanmean(abs_difference),
    "mad": np.nanmedian(np.abs(difference - np.nanmedian(difference)))
}
print(quality_dict)

import src.display.display_images as di
#di.display_images([dem_ryan, dem_ryan_resampled])
di.display_images([dem_corrected_crop, dem_ryan_resampled, difference])

print(bounds_corrected, bounds_ryan)
print(dem_corrected.shape, dem_ryan.shape)