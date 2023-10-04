import numpy as np
import rasterio

from rasterio.warp import calculate_default_transform


def check_georef_image(tiff_path, catch=True, verbose=False, pbar=None):
    # load the tiff file; we need to load the image with rasterio directly, as we need the ds
    with rasterio.open(tiff_path, 'r') as ds:

        # transform and get transformed width and height
        dst_crs = ds.crs
        dst_transform, dst_width, dst_height = calculate_default_transform(ds.crs, dst_crs, ds.width,
                                                                           ds.height,
                                                                           *ds.bounds)
        # get difference between height and width of image
        diff = np.abs(((dst_width - dst_height) / dst_height) * 100)

        # Get the pixel coordinates of the four corners of the image
        rows, cols = ds.height, ds.width
        corners = [(0, 0), (0, rows), (cols, rows), (cols, 0)]

        # Convert pixel coordinates to geographic coordinates using GCPs
        geocorners = [ds.transform * xy for xy in corners]

        # Calculate distances between corner points
        distances = [np.linalg.norm(np.array(geocorners[i]) - np.array(geocorners[(i + 1) % 4])) for i in range(4)]

        # Measure angles formed by corner points
        angles = []
        for i in range(4):
            p1 = np.array(geocorners[i])
            p2 = np.array(geocorners[(i + 1) % 4])
            p3 = np.array(geocorners[(i + 2) % 4])
            v1 = p2 - p1
            v2 = p3 - p2
            angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            angles.append(angle)

    # Define threshold values for acceptable variations (parallelogram)
    diff_height_width = 25 # percent
    max_pixel_size = 10 # in m
    angle_threshold = 10  # degrees

    # There are three options for when an image is wrong:
    # 1) Difference between height and width is too big
    if diff > diff_height_width:
        print(f"Difference too big ({diff})")
        return False

    # 2) The pixels-size is too big
    elif np.abs(dst_transform[0]) > max_pixel_size or np.abs(dst_transform[4]) > max_pixel_size:
        print(f"Pixel size too big ({dst_transform[0]}, {dst_transform[4]})")
        return False

    # 3) The image is a parallelogram and not a rectangle
    elif (any(abs(angle - 90) > angle_threshold for angle in angles)):
        print(f"Image is Parallelogramm ({angles})")
        return False

    # If image passed all three tests it is good
    else:
        return True


if __name__ == "__main__":

    import os

    tiff_path = "/data_1/ATM/data_1/playground/georef2/tiffs/georeferenced/"
    tp_img_path = "/data_1/ATM/data_1/playground/georef2/images/tie_points/"
    tp_shp_path = "/data_1/ATM/data_1/playground/georef2/shape_files/tie_points/"
    import base.connect_to_db as ctd
    _sql_string = "SELECT image_id FROM images WHERE view_direction = 'V'"
    ids = ctd.get_data_from_db(_sql_string).values.tolist()
    ids = [item for sublist in ids for item in sublist]

    import random
    random.shuffle(ids)

    for image_id in ids:
        if os.path.isfile(tiff_path + image_id + ".tif") is False:
            continue
        status = check_georef_image(tiff_path + image_id + ".tif")

        if status is False:
            os.remove(tiff_path  + image_id + ".tif")
            os.remove(tp_img_path + image_id + ".png")
            os.remove(tp_shp_path + image_id + "_points.shp")

            print(image_id, "removed")
