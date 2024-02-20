import logging
import os.path

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
DEFAULT_SAVE_FOLDER = "/data_1/ATM/data_1/georef"

# define input bounds or image ids
INPUT_TYPE = "bounds"  # can be either "bounds" or "ids
BOUNDS = [-2307556, 1002181, -2088228, 1172914]  # min_x, min_y, max_x, max_y
IMAGE_IDS = []

# which type of geo-referencing should be done
GEOREF_WITH_SATELLITE = True
GEOREF_WITH_IMAGE = False
GEOREF_WITH_CALC = False

# settings for sat
MIN_COMPLEXITY = 0.05
VERIFY_IMAGE_POSITIONS = False

# define if images of a certain type should be overwritten
OVERWRITE_SAT = False
OVERWRITE_IMG = False
OVERWRITE_CALC = False

# define if failed images should be done again
RETRY_FAILED_SAT = False
RETRY_FAILED_IMG = False
RETRY_FAILED_CALC = False

# define if invalid images should be done again
RETRY_INVALID_SAT = False
RETRY_INVALID_IMG = False
RETRY_INVALID_CALC = False

def georef(input_ids, processed_images=None,
           georef_with_satellite=True, georef_with_image=True, georef_with_calc=True,
           min_complexity=0, verify_image_positions=True,
           overwrite_sat=False, overwrite_img=False, overwrite_calc=False,
           retry_failed_sat=True, retry_failed_img=True, retry_failed_calc=True,
           retry_invalid_sat=True, retry_invalid_img=True, retry_invalid_calc=True,
           catch=False):

    # flag required for the loop
    keyboard_interrupt = False

    # init the geo-reference objects
    georef_sat = gs.GeorefSatellite(min_tps_final=25, enhance_image=False,
                                    locate_image=True, tweak_image=True, filter_outliers=True)
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
    sql_string_extracted = "SELECT image_id, text_bbox, complexity, " \
                           "ST_AsText(footprint_approx) AS footprint_approx " \
                           "FROM images_extracted"
    data_extracted = ctd.execute_sql(sql_string_extracted, conn)

    # init processed images if not given
    if processed_images is None:
        processed_images = {}

    # geo-reference with satellite
    if georef_with_satellite:

        print("Start geo-referencing with satellite")

        # iterate all images
        for image_id in tqdm(input_ids):

            # check if image is already geo-referenced
            if (processed_images.get(image_id, {}).get('method') == "sat" and
                    processed_images.get(image_id, {}).get('status') == "georeferenced" and
                    not overwrite_sat):
                print(f"{image_id} already geo-referenced")
                continue

            # try to geo-reference the image
            try:

                # ignore images with a too low complexity
                if data_extracted.loc[
                   data_extracted['image_id'] == image_id]['complexity'].iloc[0] < min_complexity:
                    processed_images[image_id] = {"method": "sat", "status": "invalid",
                                                  "reason": "complexity"}
                    continue

                print(f"Geo-reference {image_id}")

                # load the image
                image = li.load_image(image_id)

                # load the mask
                mask = _prepare_mask(image_id, image, data_fid_marks, data_extracted)

                # get the approx footprint of the image
                approx_footprint = data_extracted.loc[
                    data_extracted['image_id'] == image_id]['footprint_approx'].iloc[0]

                # get the azimuth of the image
                azimuth = data_images.loc[
                    data_images['image_id'] == image_id]['azimuth'].iloc[0]

                # get the month of the image
                month = data_images.loc[
                    data_images['image_id'] == image_id]['date_month'].iloc[0]

                # check if all required data is there
                if approx_footprint is None:
                    processed_images[image_id] = {"method": "sat", "status": "missing_data",
                                                  "reason": "approx_footprint"}
                    continue
                elif mask is None:
                    processed_images[image_id] = {"method": "sat", "status": "missing_data",
                                                  "reason": "mask"}
                    continue
                elif azimuth is None:
                    processed_images[image_id] = {"method": "sat", "status": "missing_data",
                                                  "reason": "azimuth"}
                    continue

                # we need to adapt the azimuth to account for EPSG:3031
                azimuth = 360 - azimuth + 90

                # the actual geo-referencing
                transform, residuals, tps, conf = georef_sat.georeference(image, approx_footprint,
                                                                          mask, azimuth, month)

                # skip images we can't geo-reference
                if transform is None:
                    processed_images[image_id] = {"method": "sat", "status": "failed",
                                                  "reason": "no_transform"}
                    continue

                # verify the geo-referenced image
                valid_image, reason = vig.verify_image_geometry(image, transform)

                # save valid images
                if valid_image:
                    _save_results("sat", image_id, image, transform, residuals, tps, conf, azimuth, month)
                    processed_images[image_id] = {"method": "sat", "status": "georeferenced",
                                                  "reason": ""}

                else:
                    processed_images[image_id] = {"method": "sat", "status": "invalid",
                                                  "reason": reason}

                print(f"Geo-referencing of {image_id} finished")

            # manually catch the keyboard interrupt
            except KeyboardInterrupt:
                keyboard_interrupt = True

            # that means something in the geo-referencing process fails
            except (Exception,) as e:

                processed_images[image_id] = {"method": "sat", "status": "failed",
                                              "reason": "exception"}

                if catch is False:
                    raise e

                print(f"Geo-referencing of {image_id} failed")

            # we always want to add information to the csv file
            finally:

                if keyboard_interrupt is False:
                    csv_path = DEFAULT_SAVE_FOLDER + "/processed_images.csv"
                    mc.modify_csv(csv_path, image_id, "add", processed_images[image_id], overwrite=True)

        # now that we have more images, verify the images again to check for wrong positions
        if verify_image_positions:
            for image_id in processed_images:
                # load the image
                image = li.load_image(image_id)

                valid_image = vip.verify_image_position()

                # set image to invalid if position is wrong
                if not valid_image:
                    processed_images[image_id] = {"method": "sat", "status": "invalid",
                                                  "reason": "position"}
                    mc.modify_csv(csv_path, image_id, "add", processed_images[image_id], overwrite=True)

    # geo-reference with image
    if georef_with_image:
        for image_id in tqdm(input_ids):

            # try to geo-reference the image
            try:
                # load the image
                image = li.load_image(image_id)

                # load the mask
                mask = _prepare_mask(image_id, image, data_fid_marks, data_extracted)

                georef_img.georeference(image, mask)
            except (Exception,) as e:

                if catch is False:
                    raise e

    # geo-reference with calc
    if georef_with_calc:
        for image_id in tqdm(input_ids):
            gc.georeference(image_id)


def _prepare_mask(image_id, image, data_fid_marks, data_extracted):
    # Get the fid marks for the specific image_id
    fid_marks_row = data_fid_marks.loc[data_fid_marks['image_id'] == image_id].squeeze()

    # Create fid mark dict using dictionary comprehension
    fid_dict = {i: (fid_marks_row[f'fid_mark_{i}_x'], fid_marks_row[f'fid_mark_{i}_y']) for
                i in range(1, 5)}

    # get the text boxes of the image
    text_string = data_extracted.loc[data_extracted['image_id'] == image_id]['text_bbox'].iloc[0]

    # create text-boxes list
    text_boxes = [list(group) for group in eval(text_string.replace(";", ","))]

    # load the mask
    mask = cm.create_mask(image, fid_dict, text_boxes)

    return mask


def _save_results(georef_type, image_id, image, transform, residuals, tps, conf, azimuth, month):

    # save the geo-referenced image
    path_img = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}.tif"
    at.apply_transform(image, transform, save_path=path_img)

    # save the transform
    path_transform = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}_transform.txt"
    np.savetxt(path_transform, transform)

    # save the points and conf
    path_points = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}_points.txt"
    tps_conf = np.concatenate([tps, conf.reshape(-1, 1)], axis=1)
    np.savetxt(path_points, tps_conf)

    # define attributes for shapefile
    attributes = {
        'image_id': image_id,
        'azimuth': azimuth,
        'month': month,
        'num_tps': tps.shape[0],
        'avg_conf': round(np.mean(conf), 3),
        'avg_resi': round(np.mean(residuals), 3),
    }
    attributes = pd.DataFrame.from_dict(attributes, orient='index').T

    # save footprint to shp file
    path_shp_file = f"{DEFAULT_SAVE_FOLDER}/{georef_type}.shp"
    footprint = citf.convert_image_to_footprint(image, transform)
    eg.export_geometry(footprint, path_shp_file,
                       attributes=attributes, key_field="image_id",
                       overwrite_file=False,
                       overwrite_entry=True, attach=True)


if __name__ == "__main__":

    # load image ids with bounds
    if INPUT_TYPE == "bounds":
        # load all approximate positions of images
        import src.load.load_shape_data as lsd

        path_approx_shape = "/data_1/ATM/data_1/shapefiles/TMA_Photocenters/TMA_pts_20100927.shp"
        image_positions = lsd.load_shape_data(path_approx_shape)

        # filter the ids inside the bounds
        _input_ids = ei.extract_ids(BOUNDS, image_positions,
                                    image_directions=["V"], complete_flightpaths=True)

    # image ids are provided
    else:
        _input_ids = IMAGE_IDS

    # check if there is a status csv and load it
    if os.path.isfile(DEFAULT_SAVE_FOLDER + "/processed_images.csv"):
        import pandas as pd

        _processed_images = pd.read_csv(DEFAULT_SAVE_FOLDER + "/processed_images.csv", delimiter=";")
        _processed_images.set_index('id', inplace=True)
        _processed_images = _processed_images.to_dict(orient='index');
    else:
        _processed_images = None

    # call the actual geo-referencing function
    georef(_input_ids, _processed_images,
           georef_with_satellite=GEOREF_WITH_SATELLITE,
           georef_with_image=GEOREF_WITH_IMAGE,
           georef_with_calc=GEOREF_WITH_CALC,
           min_complexity=MIN_COMPLEXITY,
           verify_image_positions=VERIFY_IMAGE_POSITIONS,
           catch=False)
