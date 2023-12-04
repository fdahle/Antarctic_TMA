import copy
import math
import numpy as np
import shapely

import base.load_image_from_file as liff
import base.remove_borders as rb

import image_georeferencing.sub.deleted.derive_image_position as dip
import image_georeferencing.sub.apply_gcps as ag

#use four corners of the polygon as gcps?

output_path = "/data_1/ATM/data_1/playground/georef3/tiffs/calc"


def georef_calc(image_id, footprint=None, angle=None, catch=True, verbose=False, pbar=None):

    tiff_path = output_path + "/" + image_id + ".tif"

    # load image
    image_with_borders = liff.load_image_from_file(image_id, catch=catch,
                                                   verbose=False, pbar=pbar)

    # get the border dimensions
    image_no_borders = rb.remove_borders(image_with_borders, image_id=image_id,
                                         catch=catch, verbose=False, pbar=pbar)

    if footprint is None or angle is None:
        _, footprint, regression_line = dip.derive_image_position(image_id, polygon_mode="exact",
                                                                  return_line=True, verbose=True)
        # we need the angle of the regression line
        x1, y1 = regression_line.coords[0]
        x2, y2 = regression_line.coords[1]
        angle_rad = math.atan2(x2 - x1, y2 - y1)
        angle = math.degrees(angle_rad)

    # convert coords to numpy array
    tps_abs = list(footprint.exterior.coords)
    tps_abs = np.asarray(tps_abs)

    # remove the double row
    _, unique_indices = np.unique(tps_abs, axis=0, return_index=True)
    tps_abs = tps_abs[np.sort(unique_indices)]

    # get the coords from the image & replace with image coords
    tps_img = copy.deepcopy(tps_abs)

    if 0 <= angle < 90:
        tps_img[tps_abs[:, 0].argsort()[:2], 0] = 0
        tps_img[tps_abs[:, 1].argsort()[:2], 1] = 0
        tps_img[tps_abs[:, 0].argsort()[-2:], 0] = image_no_borders.shape[1]
        tps_img[tps_abs[:, 1].argsort()[-2:], 1] = image_no_borders.shape[0]
    elif 90 <= angle < 180:
        tps_img[tps_abs[:, 0].argsort()[:2], 0] = 0
        tps_img[tps_abs[:, 1].argsort()[:2], 1] = image_no_borders.shape[0]
        tps_img[tps_abs[:, 0].argsort()[-2:], 0] = image_no_borders.shape[1]
        tps_img[tps_abs[:, 1].argsort()[-2:], 1] = 0
    elif 180 <= angle < 270:
        tps_img[tps_abs[:, 0].argsort()[:2], 0] = image_no_borders.shape[1]
        tps_img[tps_abs[:, 1].argsort()[:2], 1] = image_no_borders.shape[0]
        tps_img[tps_abs[:, 0].argsort()[-2:], 0] = 0
        tps_img[tps_abs[:, 1].argsort()[-2:], 1] = 0
    elif 270 <= angle < 360:
        tps_img[tps_abs[:, 0].argsort()[:2], 0] = image_no_borders.shape[1]
        tps_img[tps_abs[:, 1].argsort()[:2], 1] = 0
        tps_img[tps_abs[:, 0].argsort()[-2:], 0] = 0
        tps_img[tps_abs[:, 1].argsort()[-2:], 1] = image_no_borders.shape[0]

    tps = np.concatenate((tps_abs, tps_img), axis=1)

    transform = ag.apply_gcps(tiff_path, image_no_borders, tps, "rasterio", catch=False)

    print("Transform is:")
    print(transform)

    if transform is not None:
        return "georeferenced"


if __name__ == "__main__":

    img_id = "CA181332V0082"

    import base.connect_to_db as ctd
    sql_string = "SELECT ST_AsText(footprint) AS approx_footprint FROM images_extracted " \
                 f"WHERE image_id='{img_id}'"
    data = ctd.get_data_from_db(sql_string)
    footprint = data.iloc[0]['approx_footprint']
    footprint = shapely.from_wkt(footprint)

    # we still need the angle
    sql_string = f"SELECT azimuth FROM images WHERE image_id='{img_id}'"
    data = ctd.get_data_from_db(sql_string)
    angle = data.iloc[0]['azimuth']

    georef_calc(img_id, footprint=footprint, angle=angle)