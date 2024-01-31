import json
import os
import rasterio.mask
import rasterio.merge
import shapely

import base.print_v as p

import display.display_images

debug_show_satellite_subset = False


def load_satellite_data(bounds, satellite_type=None,
                        month=None, fallback_month=True,
                        satellite_path=None, satellite_crs=None,
                        return_transform=False,
                        return_used_images=False,
                        catch=True, verbose=False, pbar=None):
    """
    load_satellite_data(bounds, satellite_type, satellite_path, satellite_crs, return_transform, return_used_images,
                        catch, verbose, pbar):
    This function loads accept the bounds of a polygon and return the satellite data from the same coordinates as a
    np-array. If the polygon goes over different satellite-tiles, these are automatically merged.
    Args:
        bounds (list): A list containing the coordinates of the bounding box of the area of interest
            in the format [x_min, y_min, x_max, y_max].
        satellite_type (str, None): A string specifying the type of satellite data to be loaded
        month (int, None): Indicating if we want satellite images from a certain month (1-12). If none,
            we will load a compilation over all months
        fallback_month (bool, True): If True and we didn't find a satellite image for certain month, we load
            instead the compilation over all months
        satellite_path (str, None): A string specifying the path to the directory containing the satellite data
        satellite_crs (int, None): An integer specifying the CRS code of the satellite data
        return_transform (Boolean, False): A boolean indicating whether to return the
            affine transformation matrix of the cropped satellite image
        return_used_images (Boolean, False): If true, the names of the satellite images are returned
        catch (Boolean, True): If true, we catch every error that is happening and return instead None
        verbose (Boolean, False): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar, None): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        cropped (np-array): The satellite image with the extent of bounds. Usually it has three bands
            (bands x height x width)
        transform (rasterio-transform) The transform of the cropped image (describing the position, pixel-size, etc)
        used_images(list): A list with the names of the used satellite images
    """

    p.print_v(f"Start: load_satellite_data ({bounds})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder[:-4] + "/params.json") as j_file:
        json_data = json.load(j_file)

    # set default satellite type
    if satellite_type is None:
        satellite_type = json_data["satellite_type"]

    # set default satellite path
    if satellite_path is None:
        satellite_path = json_data["path_folder_satellite_data"] + "/" + satellite_type

    month_strings = [
        "0_complete", "1_jan", "2_feb", "3_mar", "4_apr", "5_may", "6_jun",
        "7_jul", "8_aug", "9_sep", "10_oct", "11_nov", "12_dec"
    ]

    if month is None:
        month = 0

    satellite_path_adapted = satellite_path + "/" + month_strings[month]

    # set default satellite crs
    if satellite_crs is None:
        satellite_crs = json_data["satellite_crs"]

    try:
        # convert bounds to shapely polygon if not already polygon
        if isinstance(bounds, shapely.geometry.base.BaseGeometry) is False:
            bounds = shapely.geometry.box(*bounds)

        # here we save the files we want to merge
        mosaic_files = []
        file_names = []

        # iterate through all image files in the satellite folder
        for file in os.listdir(satellite_path_adapted):
            if file.endswith(".tif"):

                # open the image file
                src = rasterio.open(satellite_path_adapted + "/" + file)

                # we only want the satellite images with the same crs code
                crs_code = int(src.crs['init'].split(":")[1])
                if crs_code != satellite_crs:
                    src.close()
                    continue

                # get bounding box of satellite image as shapely polygon
                sat_bounds = shapely.geometry.box(*src.bounds)

                # if the bounding boxes do not intersect, we can skip this raster file
                if bounds.intersects(sat_bounds) is False:
                    src.close()
                    continue

                file_names.append(file)
                mosaic_files.append(src)

        p.print_v(f"Merge {len(mosaic_files)} satellite images", verbose=verbose, pbar=pbar)

        if len(mosaic_files) == 0:

            # we didn't find satellite images for that month, so let's get them for the complete year
            if fallback_month and month != 0:
                p.print_v("Fallback for satellite images required!")
                cr, cr_tr, used_images = load_satellite_data(bounds, satellite_type, 0, False,
                                                             satellite_path, satellite_crs,
                                                             True, True, catch, verbose, pbar)
                if return_transform and return_used_images:
                    return cr, cr_tr, used_images
                elif return_transform:
                    return cr, cr_tr
                elif return_used_images:
                    return cr, used_images
                else:
                    return cr

            p.print_v("No satellite images were found", verbose=verbose, pbar=pbar)
            if catch:
                if return_transform and return_used_images:
                    return None, None, None
                elif return_transform or return_used_images:
                    return None, None
                else:
                    return None
            else:
                raise ValueError("No satellite images were found")

        # merge the files
        merged, transform_merged = rasterio.merge.merge(mosaic_files)

        # close the connection to the mosaic files
        for file in mosaic_files:
            file.close()

        with rasterio.io.MemoryFile() as mem_file:
            with mem_file.open(
                    driver="GTiff",
                    height=merged.shape[1],
                    width=merged.shape[2],
                    count=merged.shape[0],
                    dtype=merged.dtype,
                    transform=transform_merged,
            ) as dataset:
                dataset.write(merged)

            with mem_file.open() as dataset:
                cropped, cropped_trans = rasterio.mask.mask(dataset, [bounds], crop=True)

        if debug_show_satellite_subset:
            display.display_images.display_images(cropped)
    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: load_satellite_data ({bounds})", verbose=verbose, pbar=pbar)
            if return_transform and return_used_images:
                return None, None, None
            elif return_transform or return_used_images:
                return None, None
            else:
                return None
        else:
            raise e

    p.print_v(f"Finished: load_satellite_data ({bounds})", verbose=verbose, pbar=pbar)

    if return_transform:
        if return_used_images:
            return cropped, cropped_trans, file_names
        else:
            return cropped, cropped_trans
    else:
        if return_used_images:
            return cropped, file_names
        else:
            return cropped


if __name__ == "__main__":
    _bounds = [-2350080.0, 849920.0, -2299900.0, 900100.0]

    load_satellite_data(_bounds, verbose=True)
