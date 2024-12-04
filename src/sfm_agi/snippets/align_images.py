
# Library imports
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from rasterio.env import Env

def align_images(image1_path, image2_path, output1_path, output2_path):
    """
    Aligns two georeferenced images by resampling them to a common minimal bounding box.

    Args:
        image1_path (str): Path to the first georeferenced image.
        image2_path (str): Path to the second georeferenced image.
        output1_path (str): Path to save the aligned first image.
        output2_path (str): Path to save the aligned second image.

    Returns:
        None
    """
    with Env(GTIFF_SRS_SOURCE='EPSG'):
        with rasterio.open(image1_path) as src1, rasterio.open(image2_path) as src2:
            # Extract metadata
            transform1, transform2 = src1.transform, src2.transform
            crs1, crs2 = src1.crs, src2.crs

            # Ensure both images have the same CRS
            if crs1 != crs2:
                raise ValueError("Both images must have the same CRS.")

            # Compute bounds for each image
            bounds1 = rasterio.transform.array_bounds(src1.height, src1.width, transform1)
            bounds2 = rasterio.transform.array_bounds(src2.height, src2.width, transform2)

            # Determine the overlapping bounding box
            min_x = max(bounds1[0], bounds2[0])
            min_y = max(bounds1[1], bounds2[1])
            max_x = min(bounds1[2], bounds2[2])
            max_y = min(bounds1[3], bounds2[3])

            if min_x >= max_x or min_y >= max_y:
                raise ValueError("Images do not overlap.")

            # Compute pixel size (assume same pixel size for both images)
            pixel_size_x = transform1.a
            pixel_size_y = -transform1.e

            # Compute new dimensions and transform
            width = int((max_x - min_x) / pixel_size_x)
            height = int((max_y - min_y) / pixel_size_y)
            new_transform = from_origin(min_x, max_y, pixel_size_x, pixel_size_y)

            print(src1.nodata, src2.nodata)

            # Create output arrays with the new dimensions
            aligned_image1 = np.full((height, width), src1.nodata or 0, dtype=src1.dtypes[0])
            aligned_image2 = np.full((height, width), src2.nodata or 0, dtype=src2.dtypes[0])

            # Reproject and align the first image
            rasterio.warp.reproject(
                source=rasterio.band(src1, 1),
                destination=aligned_image1,
                src_transform=transform1,
                src_crs=crs1,
                dst_transform=new_transform,
                dst_crs=crs1,
                resampling=Resampling.nearest,
            )

            # Reproject and align the second image
            reproject(
                source=rasterio.band(src2, 1),
                destination=aligned_image2,
                src_transform=transform2,
                src_crs=crs2,
                dst_transform=new_transform,
                dst_crs=crs2,
                resampling=Resampling.nearest,
            )

            # Save the aligned images
            metadata = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': src1.dtypes[0],
                'crs': crs1,
                'transform': new_transform,
                'nodata': src1.nodata,
                'compress': 'lzw',
                'BIGTIFF': 'YES',
            }

            with rasterio.open(output1_path, 'w', **metadata) as dst1:
                dst1.write(aligned_image1, 1)

            metadata['dtype'] = src2.dtypes[0]
            metadata['nodata'] = src2.nodata
            with rasterio.open(output2_path, 'w', **metadata) as dst2:
                dst2.write(aligned_image2, 1)

if __name__ == "__main__":

    input_path_1 = "/data/ATM/data_1/sfm/agi_projects/crane/output/crane_ortho_absolute.tif"
    input_path_2 = "/data/ATM/data_1/sfm/agi_projects/crane/output/crane_dem_absolute.tif"
    output_path_1 = "/data/ATM/data_1/sfm/agi_projects/crane/output/crane_ortho_aligned.tif"
    output_path_2 = "/data/ATM/data_1/sfm/agi_projects/crane/output/crane_dem_aligned.tif"

    align_images(input_path_1, input_path_2, output_path_1, output_path_2)