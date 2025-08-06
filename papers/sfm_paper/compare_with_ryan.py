project_name = "crane_glacier"

import os
import glob
import numpy as np

path_corrected_dem = "/data/ATM/data_1/papers/paper_sfm/finished_dems_corrected2/crane_glacier_dem_corrected.tif"
glac_name = project_name.split("_")[0].capitalize()
pth_ryan_dem = f"/data/ATM/data_1/ryan_data/hist_dems/North_and_Barrows_2024_DEM_{glac_name}_1968.tif"

# load our dem
import src.load.load_image as li
dem_corrected, transform_corrected = li.load_image(path_corrected_dem, return_transform=True)
dem_corrected[dem_corrected == -9999] = np.nan  # set nodata to nan

# load dem of ryan
glac_name = project_name.split("_")[0].capitalize()
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

# 4. Estimate horizontal shift via 2D cross-correlation
# remove mean and mask nan
A = dem_corrected_crop.copy()
B = dem_ryan_resampled.copy()
mask = ~np.isnan(A) & ~np.isnan(B)
A[~mask] = 0
B[~mask] = 0
A -= np.nanmean(A)
B -= np.nanmean(B)

from scipy.signal import fftconvolve

corr = fftconvolve(A, B[::-1, ::-1], mode='same')
iy, ix = np.unravel_index(np.nanargmax(corr), corr.shape)
dy = iy - A.shape[0]//2  # row shift
dx = ix - A.shape[1]//2  # col shift
print(f"Estimated shift: dx={dx} px, dy={dy} px")

# 5. Apply pixel shift to Ryan DEM
# shift in numpy (roll) or adjust transform
ryan_shifted = np.roll(dem_ryan_resampled, shift=(int(dy), int(dx)), axis=(0, 1))
# set newly rolled-in edges to nan
if dy > 0:
    ryan_shifted[:int(dy), :] = np.nan
elif dy < 0:
    ryan_shifted[int(dy):, :] = np.nan
if dx > 0:
    ryan_shifted[:, :int(dx)] = np.nan
elif dx < 0:
    ryan_shifted[:, int(dx):] = np.nan


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