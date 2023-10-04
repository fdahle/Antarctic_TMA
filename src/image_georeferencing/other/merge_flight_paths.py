import os
import numpy as np
import rasterio

from rasterio.warp import reproject, Resampling
from tqdm import tqdm

input_folder = "/data_1/ATM/data_1/playground/georef4/low_res/images"
output_folder = "/data_1/ATM/data_1/playground/georef4/low_res/flight_paths"

overwrite = True

def merge_flight_paths(input_fld, output_fld, nodata=0):
    def _get_combined_bounds_and_avg_pixel_size(files):
        """Get the combined bounds and average pixel size of a list of raster files."""
        min_left = float('inf')
        max_right = float('-inf')
        min_bottom = float('inf')
        max_top = float('-inf')

        total_x_res = 0
        total_y_res = 0

        for file in files:
            with rasterio.open(file) as src:
                left, bottom, right, top = src.bounds
                x_res, y_res = src.res

                if left < min_left:
                    min_left = left
                if right > max_right:
                    max_right = right
                if bottom < min_bottom:
                    min_bottom = bottom
                if top > max_top:
                    max_top = top

                total_x_res += x_res
                total_y_res += y_res

        avg_x_res = total_x_res / len(files)
        avg_y_res = total_y_res / len(files)

        return (min_left, min_bottom, max_right, max_top), (avg_x_res, avg_y_res)

    # List all files in the folder & sort them
    all_files = [f for f in os.listdir(input_fld) if f.endswith('.tif')]
    all_files.sort()

    # Extract unique flight paths from filenames
    flight_paths = sorted(list(set([f[2:6] for f in all_files])))

    # iterate every flight path
    for flight_path in tqdm(flight_paths):

        print(f"Calc {flight_path}")

        output_file = output_fld + "/" + flight_path + ".tif"

        if overwrite is False and os.path.isfile(output_file):
            continue

        # get all tiffs from one flight path
        tiffs_to_merge = sorted([os.path.join(input_folder, f) for f in all_files if f.startswith(f"CA{flight_path}")],
                                key=lambda x: int(x.split('V')[-1].split('.')[0]), reverse=True)

        bounds, avg_res = _get_combined_bounds_and_avg_pixel_size(tiffs_to_merge)

        min_left = bounds[0]
        min_bottom = bounds[1]
        max_right = bounds[2]
        max_top = bounds[3]

        avg_x_res = avg_res[0]
        avg_y_res = avg_res[1]

        # Define the metadata for our canvas based on the first image (assuming similar metadata across all files)
        with rasterio.open(tiffs_to_merge[0]) as src:
            meta = src.meta

        try:
            # Compute the width and height of the combined raster based on average pixel size
            width = int((max_right - min_left) / avg_x_res)
            height = int((max_top - min_bottom) / avg_y_res)

            # Update metadata for the merged raster
            meta.update(width=width, height=height,
                        transform=rasterio.transform.from_bounds(min_left, min_bottom, max_right, max_top, width,
                                                                 height))

            # Create an empty canvas to hold the merged image
            mosaic = np.zeros((height, width), dtype=np.uint8)
        except:
            continue

        # Reproject each file onto the canvas
        for file in sorted(tiffs_to_merge, key=lambda x: int(x.split('V')[-1].split('.')[0]), reverse=True):
            with rasterio.open(file) as src:

                # Reproject the image to match the final mosaic's resolution and bounds
                temp_data = np.full((height, width), nodata, dtype=np.uint8)

                reproject(
                    source=rasterio.band(src, 1),
                    destination=temp_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=meta['transform'],
                    dst_crs=meta['crs'],
                    resampling=Resampling.nearest
                )

                # Overlay the reprojected image on the mosaic. Only update pixels that are not nodata.
                valid_data = temp_data != nodata
                mosaic[valid_data] = temp_data[valid_data]

        # Write the merged image
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(mosaic, 1)


def inspect_tiff_metadata(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.tif')]
    for file in files:
        with rasterio.open(file) as src:
            print(f"File: {file}")
            print("Bounds:", src.bounds)
            print("Transform:", src.transform)
            print("------")
        exit()


if __name__ == "__main__":
    # inspect_tiff_metadata(input_folder)

    merge_flight_paths(input_folder, output_folder)
