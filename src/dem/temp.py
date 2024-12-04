import os
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import numpy as np

def temp(tiff1_path, tiff2_path, output_mask_path):
    """
    Align two rasters with slightly different origins and calculate a mask where
    the second raster is closer to zero than the first raster.

    Parameters:
    tiff1_path (str): Path to the first raster (original differences).
    tiff2_path (str): Path to the second raster (corrected differences).
    output_mask_path (str): Path to save the output mask raster.

    Returns:
    None
    """
    # Open both rasters
    with rasterio.open(tiff1_path) as src1, rasterio.open(tiff2_path) as src2:
        # Ensure the rasters have the same CRS
        if src1.crs != src2.crs:
            raise ValueError("Rasters must have the same CRS.")

        # Read metadata and calculate target transform and dimensions
        transform, width, height = calculate_default_transform(
            src1.crs, src1.crs, src1.width, src1.height, *src1.bounds
        )
        metadata = src1.meta.copy()
        metadata.update({
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": rasterio.uint8,
            "nodata": 255  # Use 255 as the nodata value for uint8
        })

        # Reproject tiff1 to the target grid
        tiff1_aligned = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src1, 1),
            destination=tiff1_aligned,
            src_transform=src1.transform,
            src_crs=src1.crs,
            dst_transform=transform,
            dst_crs=src1.crs,
            resampling=Resampling.bilinear
        )

        # Reproject tiff2 to the target grid
        tiff2_aligned = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src2, 1),
            destination=tiff2_aligned,
            src_transform=src2.transform,
            src_crs=src2.crs,
            dst_transform=transform,
            dst_crs=src2.crs,
            resampling=Resampling.bilinear
        )

        # Calculate mask where tiff2_aligned is closer to zero
        mask = (np.abs(tiff2_aligned) < np.abs(tiff1_aligned)).astype(np.uint8)

        # Assign nodata value to areas that should be excluded
        mask[np.isnan(tiff1_aligned)] = 255
        mask[np.isnan(tiff2_aligned)] = 255

        # Save the mask to a new raster
        with rasterio.open(output_mask_path, "w", **metadata) as dst:
            dst.write(mask, 1)

# Define input and output paths
project_name = "test3"
project_folder = f"/data/ATM/data_1/sfm/agi_projects/{project_name}"
output_folder = os.path.join(project_folder, "output")
corrected_dem = os.path.join(output_folder, project_name + "_diff_rel_corrected.tif")
original_dem = os.path.join(output_folder, project_name + "_diff_rel.tif")
output_dem = os.path.join(output_folder, project_name + "temp.tif")

# Run the function
temp(corrected_dem, original_dem, output_dem)
