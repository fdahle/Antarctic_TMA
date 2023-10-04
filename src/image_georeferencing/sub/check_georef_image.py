import numpy as np
import rasterio

from rasterio.warp import calculate_default_transform

import base.print_v as p


def check_georef_image(tiff_path, return_reason=False,
                       catch=True, verbose=False, pbar=None):
    """
    check_georef_image(tiff_path, return_reason, catch, verbose, pbar):
    This function looks at a geo-referenced image and checks its validity for different criteria. Currently following
    criteria are supported:
    - difference between image width and height (images should be square)
    - pixel size of the image (the geo-referenced images all should have pixels with a certain size)
    - angle of the image (The image should have angles of around 90degrees and not be a parallelogram)
    Args:
        tiff_path (String): The path to the geo-referenced image
        return_reason (Boolean, False): If true, the reason why an image failed is returned
        catch (Boolean, True): If true and something is going wrong, the operation will continue and not crash.
            In this case None is returned
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        validity (Boolean): True if the image is valid, false if the image is invalid
        reason (String, optional): An empty String if the image is valid, otherwise the reason for invalidity
    """
    try:
        # load the tiff file; we need to load the image with rasterio directly, as we need the ds
        with rasterio.open(tiff_path, 'r') as ds:

            # transform and get transformed width and height
            dst_crs = ds.crs
            dst_transform, dst_width, dst_height = calculate_default_transform(ds.crs, dst_crs, ds.width,
                                                                               ds.height,
                                                                               *ds.bounds)

            # Get the pixel coordinates of the four corners of the image
            rows, cols = ds.height, ds.width
            corners = [(0, 0), (0, rows), (cols, rows), (cols, 0)]

            # Convert pixel coordinates to geographic coordinates using GCPs
            geo_corners = [ds.transform * xy for xy in corners]

            # Calculate distances between corner points
            distances = [np.linalg.norm(np.array(geo_corners[i]) -
                                        np.array(geo_corners[(i + 1) % 4])) for i in range(4)]

            # get difference between height and width of image
            diff = np.abs(((distances[0] - distances[1]) / distances[1]) * 100)

            # Measure angles formed by corner points
            angles = []
            for i in range(4):
                p1 = np.array(geo_corners[i])
                p2 = np.array(geo_corners[(i + 1) % 4])
                p3 = np.array(geo_corners[(i + 2) % 4])
                v1 = p2 - p1
                v2 = p3 - p2
                angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
                angles.append(angle)

        # Define threshold values for acceptable variations
        diff_height_width = 15  # percent
        max_pixel_size = 2  # in m
        angle_threshold = 10  # degrees

        # There are three options for when an image is wrong:
        # 1) Difference between height and width is too big
        if diff > diff_height_width:

            diff = round(diff, 2)

            p.print_v(f"Difference too big ({diff})", verbose=verbose, pbar=pbar)
            if return_reason:
                return False, f"diff_to_big({diff})"
            else:
                return False

        # 2) The pixels-size is too big
        elif np.abs(dst_transform[0]) > max_pixel_size or np.abs(dst_transform[4]) > max_pixel_size:
            pix_x = round(dst_transform[0], 4)
            pix_y = round(dst_transform[4], 4)
            p.print_v(f"Pixel size too big ({pix_x}, {pix_y})", verbose=verbose, pbar=pbar)
            if return_reason:
                return False, f"pixel_size_too_big({pix_x},{pix_y})"
            else:
                return False

        # 3) The image is a parallelogram and not a rectangle
        elif any(abs(angle - 90) > angle_threshold for angle in angles):
            p.print_v(f"Image is Parallelogram ({angles})", verbose=verbose, pbar=pbar)
            if return_reason:
                return False, "parallelogram"
            else:
                return False

        # If image passed all three tests it is good
        else:
            if return_reason:
                return True, ""
            else:
                return True

    except (Exception,) as e:
        if catch:
            if return_reason:
                return None, None
            else:
                return None
        else:
            raise e


if __name__ == "__main__":

    import os

    _tiff_path = "/data_1/ATM/data_1/playground/georef3/tiffs/sat/"
    _tp_img_path = "/data_1/ATM/data_1/playground/georef3/tie_points/images/sat"
    _tp_shp_path = "/data_1/ATM/data_1/playground/georef2/tie_points/shapes/sat"

    import base.connect_to_db as ctd
    _sql_string = "SELECT image_id FROM images WHERE view_direction = 'V'"
    _ids = ctd.get_data_from_db(_sql_string).values.tolist()
    _ids = [item for sublist in _ids for item in sublist]

    import random
    random.shuffle(_ids)

    _ids = ["CA181732V0015"]

    for _image_id in _ids:
        if os.path.isfile(_tiff_path + _image_id + ".tif") is False:
            print(f"{_image_id} not existing at '{_tiff_path}{_image_id}.tif'")
            continue
        status = check_georef_image(_tiff_path + _image_id + ".tif", verbose=True)

        # if status is False:
        #    os.remove(_tiff_path  + _image_id + ".tif")
        #    os.remove(_tp_img_path + _image_id + ".png")
        #    os.remove(_tp_shp_path + _image_id + "_points.shp")

        #    print(_image_id, "removed")
