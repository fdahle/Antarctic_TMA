import logging

import numpy as np
from tqdm import tqdm

# import base functions
import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.base.modify_csv as mc

# import export functions
import src.export.export_geometry as eg

# import extract function
import src.extract.extract_ids as ei

# import load functions
import src.load.load_image as li

# import georef functions
import src.georef.georef_sat as gs
import src.georef.georef_img as gi
import src.georef.georef_calc as gc

# import georef snippet functions
import src.georef.snippets.apply_transform as at
import src.georef.snippets.convert_image_to_footprint as citf
import src.georef.snippets.verify_image_geometry as vig
import src.georef.snippets.verify_image_position as vip

# define save folder
DEFAULT_SAVE_FOLDER = "/data_1/ATM/data_1/georef/"

# define input bounds or image ids
INPUT_TYPE = "ids"  # can be either "bounds" or "ids
BOUNDS = [613566, 2530845, 1318654, 1965841]
IMAGE_IDS = []

GEOREF_WITH_SATELLITE = True
GEOREF_WITH_IMAGE = False
GEOREF_WITH_CALC = False


def georef(input_ids, processed_images={},
           georef_with_satellite=True, georef_with_image=True, georef_with_calc=True,
           catch=False):

    # init the geo-reference objects
    georef_sat = gs.GeorefSatellite()
    georef_img = gi.GeorefImage()
    georef_calc = gc.GeorefCalc()

    # establish connection to the database
    conn = ctd.establish_connection()

    # get month and angles from the database
    sql_string_images = "SELECT image_id, azimuth, date_month FROM images"
    data_images = ctd.execute_sql(sql_string_images, conn)

    # get fid mark data
    sql_string_fid_marks = "SELECT * FROM images_fid_points"
    data_fid_marks = ctd.execute_sql(sql_string_fid_marks, conn)

    # get extracted data
    sql_string_extracted = "SELECT image_id, text_bbox, ST_AsText(footprint_approx) AS footprint_approx " \
                           "FROM images_extracted"
    data_extracted = ctd.execute_sql(sql_string_extracted, conn)

    # here we save during runtime the images and their status
    processed_images = {}

    # geo-reference with satellite
    if georef_with_satellite:
        for image_id in tqdm(input_ids):

            try:
                # load the image
                image = li.load_image(image_id)

                # load the mask
                mask = _prepare_mask(image_id, image, data_fid_marks, data_extracted)

                # get the approx footprint of the image
                approx_footprint = data_extracted.loc[data_extracted['image_id'] == image_id]['approx_footprint']

                # get the azimuth of the image
                azimuth = data_images.loc[data_images['image_id'] == image_id]['angle']

                # get the month of the image
                month = data_images.loc[data_images['image_id'] == image_id]['date_month']

                # check if all required data is there
                if approx_footprint is None:
                    processed_images[image_id] = {"status": "missing_data", "reason": "approx_footprint"}
                    continue
                elif mask is None:
                    processed_images[image_id] = {"status": "missing_data", "reason": "mask"}
                    continue
                elif azimuth is None:
                    processed_images[image_id] = {"status": "missing_data", "reason": "azimuth"}
                    continue

                # we need to adapt the azimuth to account for EPSG:3031
                azimuth = 360 - azimuth + 90

                # the actual geo-referencing
                transform, residuals, tps, conf = georef_sat.georeference(image, approx_footprint,
                                                                          mask, azimuth, month)

                # skip images we can't geo-reference
                if transform is None:
                    processed_images[image_id] = {"status": "failed", "reason": "satellite"}
                    continue

                # verify the geo-referenced image
                valid_image, reason = vig.verify_image_geometry(image, transform)

                # save valid images
                if valid_image:
                    _save_results("sat", image_id, image, transform, residuals, tps, conf)
                    processed_images[image_id] = {"status": "georeferenced", "reason": "satellite"}
                else:
                    processed_images[image_id] = {"status": "invalid", "reason": reason}

            # that means something in the geo-referencing process fails
            except (Exception,) as e:
                processed_images[image_id] = {"status": "failed", "reason": "exception"}

                if catch is False:
                    raise e

            # we always want to add information to the csv file
            finally:

                csv_path = DEFAULT_SAVE_FOLDER + "/processed_images.csv"
                mc.modify_csv(csv_path, image_id, "add", processed_images[image_id], overwrite=True)

        # now that we have more images, verify the images again to check for wrong positions
        for image_id in processed_images:
            # load the image
            image = li.load_image(image_id)

            vip.verify_image_position()

    # geo-reference with image
    if georef_with_image:
        for image_id in tqdm(input_ids):

            # load the image
            image = li.load_image(image_id)

            # load the mask
            mask = _prepare_mask(image_id, image, data_fid_marks, data_extracted)

            georef_img.georeference(image, mask)

    # geo-reference with calc
    if georef_with_calc:
        for image_id in tqdm(input_ids):
            gc.georeference(image_id)


def _prepare_mask(image_id, image, data_fid_marks, data_extracted):
    # Get the fid marks for the specific image_id
    fid_marks_row = data_fid_marks.loc[data_fid_marks['image_id'] == image_id].squeeze()

    # Create fid mark dict using dictionary comprehension
    fid_dict = {i: (fid_marks_row[f'fid_mark_{i}_x'], fid_marks_row[f'fid_mark_{i}_y']) for i in range(1, 5)}

    # get the text boxes of the image
    text_string = data_extracted.loc[data_extracted['image_id'] == image_id]['text_bbox']

    # create text-boxes list
    text_boxes = [list(group) for group in eval(text_string.replace(";", ","))]

    # load the mask
    mask = cm.create_mask(image, fid_dict, text_boxes)

    return mask


def _save_results(georef_type, image_id, image, transform, residuals, tps, conf):
    # define the save paths
    path_img = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}.tif"
    path_transform = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}_transform.txt"
    path_points = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}_points.txt"

    # define the path for the overview-shp file
    path_shp_file = f"{DEFAULT_SAVE_FOLDER}/{georef_type}.shp"

    # create a footprint for this image
    footprint = citf.convert_image_to_footprint(image, transform)

    # save footprint to shp file
    eg.export_geometry(footprint, path_shp_file)

    # save the geo-referenced image
    at.apply_transform(image, transform, save_path=path_img)

    # save the transform
    np.savetxt(path_transform, transform)

    # save the points and conf
    tps_conf = np.concatenate([tps, conf], axis=1)
    np.save(path_points, tps_conf)


if __name__ == "__main__":

    if INPUT_TYPE == "bounds":
        # load all approximate positions of images
        import src.load.load_shape_data as lsd

        path_approx_shape = "/data_1/ATM/data_1/shapefiles/TMA_Photocenters/TMA_pts_20100927.shp"
        image_positions = lsd.load_shape_data(path_approx_shape)

        # filter the ids inside the bounds
        input_ids = ei.extract_ids(BOUNDS, image_positions,
                                   image_directions=["V"], complete_flightpaths=True)

    else:
        input_ids = IMAGE_IDS

    print(input_ids)

    # call the actual geo-referencing function
    georef(input_ids,
           georef_with_satellite=GEOREF_WITH_SATELLITE,
           georef_with_image=GEOREF_WITH_IMAGE,
           georef_with_calc=GEOREF_WITH_CALC, catch=False)
