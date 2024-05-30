# Package imports
import hashlib
import numpy as np
import os
import pandas as pd
import shutil
import time
import torch.cuda
from datetime import datetime
from tqdm import tqdm
from shapely.wkt import loads

# import base functions
import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.base.modify_csv as mc

# import export functions
import src.export.export_geometry as eg

# import georef functions
import src.georef.georef_sat as gs
import src.georef.georef_calc as gc

# import georef snippet functions
import src.georef.snippets.apply_transform as at
import src.georef.snippets.calc_azimuth as ca
import src.georef.snippets.convert_image_to_footprint as citf
import src.georef.snippets.verify_image_geometry as vig
import src.georef.snippets.verify_image_position as vip

# import load functions
import src.load.load_image as li
import src.load.load_shape_data as lsd

# define save folder
DEFAULT_SAVE_FOLDER = "/data_1/ATM/data_1/georef"

# settings for adapted georef
MIN_COMPLEXITY = 0.05
VERIFY_IMAGE_POSITIONS = True
MIN_GEOREF_SRC = "sat"  # can be "sat", "img", "calc"

# settings for verifying
DISTANCE_THRESHOLD = 100  # TODO UPDATE THIS!

# define if we should overwrite
OVERWRITE_ADAPTED = False

RETRY_MISSING_ADAPTED = False
RETRY_FAILED_ADAPTED = False
RETRY_INVALID_ADAPTED = False


def georef_adapted(input_ids,
                   overwrite_adapted=False, retry_missing_adapted=False,
                   retry_failed_adapted=False, retry_invalid_adapted=False,
                   min_complexity=0.0, verify_image_positions=True, distance_threshold=100,
                   catch=False):
    # flag required for the loop
    keyboard_interrupt = False

    georef_sat = gs.GeorefSatellite(min_tps_final=25, enhance_image=False,
                                    locate_image=False, tweak_image=True, filter_outliers=True)
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

    # get existing footprints
    sql_string_georef = "SELECT image_id, georef_type, " \
                        "ST_AsText(footprint_exact) AS footprint_exact FROM images_georef_3"
    data_georef = ctd.execute_sql(sql_string_georef, conn)

    if MIN_GEOREF_SRC == "sat":
        data_georef = data_georef[data_georef['georef_type'] == "sat"]
    elif MIN_GEOREF_SRC == "img":
        data_georef = data_georef[data_georef['georef_type'] == "sat" or
                                  data_georef['georef_type'] == "img"]
    elif MIN_GEOREF_SRC == "calc":
        # do nothing
        pass

    # set the path to the csv files with processed images & load them
    csv_path_sat = DEFAULT_SAVE_FOLDER + "/sat_processed_images.csv"
    processed_images_sat = _load_processed_images(csv_path_sat)

    # get also csv information from adapted images
    csv_path_adapted = DEFAULT_SAVE_FOLDER + "/adapted_processed_images.csv"
    processed_images_adapted = _load_processed_images(csv_path_adapted)

    print("Start geo-referencing with satellite adapted")

    # iterate all images
    for img_counter, image_id in enumerate(tqdm(input_ids)):

        # check if image is already geo-referenced
        if processed_images_sat.get(image_id, {}).get('status') == "georeferenced":
            print(f"{image_id} already geo-referenced with satellite")
            continue

        # check if image is already geo-referenced with adapted method
        if (processed_images_adapted.get(image_id, {}).get('status') == "georeferenced" and
                not overwrite_adapted):
            print(f"{image_id} already geo-referenced with adapted method")
            continue

        # check if image is already invalid
        if (processed_images_adapted.get(image_id, {}).get('status') == "invalid" and
                not retry_invalid_adapted):
            print(f"{image_id} already invalid")
            continue

        # check if image did already fail
        if (processed_images_adapted.get(image_id, {}).get('status') == "failed" and
                not retry_failed_adapted):
            print(f"{image_id} already failed")
            continue

        # check if image was missing data
        if (processed_images_adapted.get(image_id, {}).get('status') == "missing_data" and
                not retry_missing_adapted):
            print(f"{image_id} already missing data")
            continue

        # backup the processed_images every 10th iteration
        if img_counter % 10 == 0:  # noqa
            source_directory = os.path.dirname(csv_path_sat)
            backup_directory = os.path.join(source_directory, 'backup')
            filename = os.path.basename(csv_path_sat)
            backup_path = os.path.join(backup_directory, filename)
            shutil.copy(csv_path_sat, backup_path)

            source_directory = os.path.dirname(csv_path_adapted)
            backup_directory = os.path.join(source_directory, 'backup')
            filename = os.path.basename(csv_path_adapted)
            backup_path = os.path.join(backup_directory, filename)
            shutil.copy(csv_path_sat, backup_path)

        # start the timer
        start = time.time()

        # init variable
        likely_hash = ""

        # try to geo-reference the image
        try:
            # get datetime
            now = datetime.now()
            date_time_str = now.strftime("%d.%m.%Y %H:%M")

            # ignore images with a too low complexity
            if data_extracted.loc[data_extracted['image_id'] == image_id]['complexity'].iloc[0] < min_complexity:
                processed_images_adapted[image_id] = {"method": "adapted", "status": "failed",
                                                      "poly_hash": "",
                                                      "reason": "complexity", "date": date_time_str}
                print(f"{image_id} has low complexity")
                continue

            # load the image
            image = li.load_image(image_id, catch=True)

            if image is None:
                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                processed_images_adapted[image_id] = {"method": "adapted", "status": "failed",
                                                      "poly_hash": "",
                                                      "reason": "image", "date": date_time_str}
                print(f"{image_id} could not be loaded")
                continue

            # load the mask
            mask = _prepare_mask(image_id, image, data_fid_marks, data_extracted)

            # get the geo-referenced ids and footprints for that image from that flight_path
            data_fl = data_georef[data_georef['image_id'].apply(lambda x: x[2:6] == image_id[2:6])]
            fl_ids = data_fl['image_id'].tolist()
            fl_footprints = data_fl['footprint_exact'].tolist()
            fl_footprints = [loads(wkt) for wkt in fl_footprints]

            # get the likely footprint of the image
            likely_transform, _, _, _ = georef_calc.georeference(image, image_id,
                                                                 fl_ids, fl_footprints)
            likely_footprint = citf.convert_image_to_footprint(image, likely_transform, catch=True)

            # create hash for that footprint
            if likely_footprint is not None:
                likely_hash = hashlib.sha256(likely_footprint.wkt.encode('utf-8')).hexdigest()

                if processed_images_adapted.get(image_id, {}).get('poly_hash') == likely_hash:
                    print(f"{image_id} already tried to geo-reference with this footprint")
                    continue

            # get the likely azimuth of the image
            likely_azimuth = ca.calc_azimuth(image_id, conn)

            # get the month of the image
            month = data_images.loc[
                data_images['image_id'] == image_id]['date_month'].iloc[0]

            # get datetime
            now = datetime.now()
            date_time_str = now.strftime("%d.%m.%Y %H:%M")

            # check if all required data is there
            if likely_footprint is None:
                processed_images_adapted[image_id] = {"method": "adapted", "status": "missing_data",
                                                      "reason": "likely_footprint",
                                                      "poly_hash": likely_hash,
                                                      "time": "", "date": date_time_str}
                print(f"{image_id} has no approx_footprint")
                continue
            elif mask is None:
                processed_images_adapted[image_id] = {"method": "adapted", "status": "missing_data",
                                                      "reason": "mask",
                                                      "poly_hash": likely_hash,
                                                      "time": "", "date": date_time_str}
                print(f"{image_id} has no mask")
                continue
            elif likely_azimuth is None:
                processed_images_adapted[image_id] = {"method": "adapted", "status": "missing_data",
                                                      "reason": "azimuth",
                                                      "poly_hash": likely_hash,
                                                      "time": "", "date": date_time_str}
                print(f"{image_id} has no azimuth")
                continue

            print(f"Geo-reference {image_id} with satellite")

            # we need to adapt the azimuth to account for EPSG:3031
            # azimuth = 360 - azimuth + 90
            azimuth = likely_azimuth

            # the actual geo-referencing
            transform, residuals, tps, conf = georef_sat.georeference(image, likely_footprint,
                                                                      mask, azimuth, month)

            # skip images we can't geo-reference
            if transform is None:

                georef_time = round(time.time() - start)

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                # images failed due to exception
                if tps is None:
                    processed_images_adapted[image_id] = {"method": "adapted", "status": "failed",
                                                          "reason": "exception",
                                                          "poly_hash": likely_hash,
                                                          "time": georef_time, "date": date_time_str}
                # too few tps
                else:
                    processed_images_adapted[image_id] = {"method": "adapted", "status": "failed",
                                                          "reason": f"too_few_tps:{tps.shape[0]}",
                                                          "poly_hash": likely_hash,
                                                          "time": georef_time, "date": date_time_str}
                continue

            # verify the geo-referenced image
            valid_image, reason = vig.verify_image_geometry(image, transform)

            georef_time = round(time.time() - start)

            # get datetime
            now = datetime.now()
            date_time_str = now.strftime("%d.%m.%Y %H:%M")

            # save valid images
            if valid_image:

                _save_results("sat", image_id, image, transform, residuals, tps, conf, month)
                processed_images_adapted[image_id] = {"method": "adapted", "status": "georeferenced",
                                                      "reason": "",
                                                      "poly_hash": likely_hash,
                                                      "time": georef_time, "date": date_time_str}

            # set image to invalid if position is wrong
            else:
                processed_images_adapted[image_id] = {"method": "adapted", "status": "invalid",
                                                      "reason": reason,
                                                      "poly_hash": likely_hash,
                                                      "time": georef_time, "date": date_time_str}

            print(f"Geo-referencing of {image_id} finished")

        # manually catch the keyboard interrupt
        except KeyboardInterrupt:
            keyboard_interrupt = True

        # that means something in the geo-referencing process fails
        except (Exception,) as e:

            # get datetime
            now = datetime.now()
            date_time_str = now.strftime("%d.%m.%Y %H:%M")

            processed_images_adapted[image_id] = {"method": "adapted", "status": "failed",
                                                  "reason": "exception",
                                                  "poly_hash": likely_hash,
                                                  "time": round(time.time() - start),
                                                  "date": date_time_str}

            if catch is False:
                raise e

            print(f"Geo-referencing of {image_id} failed")

        # we always want to add information to the csv file
        finally:

            if keyboard_interrupt is False:
                mc.modify_csv(csv_path_adapted, image_id, "add", processed_images_adapted[image_id], overwrite=True)

            # clear the memory
            torch.cuda.empty_cache()

    # now that we have more images, verify the images again to check for wrong positions
    if verify_image_positions:

        # load footprint and ids of images that are already geo-referenced by satellite
        path_sat_shapefile = "/data_1/ATM/data_1/georef/sat.shp"
        sat_shape_data = lsd.load_shape_data(path_sat_shapefile)

        # iterate all images
        for image_id in tqdm(processed_images_adapted):

            footprint = sat_shape_data.loc[
                sat_shape_data['image_id'] == image_id].geometry.iloc[0]
            line_footprints = sat_shape_data.loc[
                sat_shape_data['image_id'][2:6] == image_id[2:6]].geometry

            valid_image = vip.verify_image_position(footprint, line_footprints, distance_threshold)

            # set image to invalid if position is wrong
            if not valid_image:
                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                # TODO FIX TIMER
                processed_images_adapted[image_id] = {"method": "adapted", "status": "invalid",
                                                      "reason": "position",
                                                      "date": date_time_str}
                mc.modify_csv(csv_path_adapted, image_id, "add", processed_images_adapted[image_id], overwrite=True)


def _prepare_mask(image_id: str, image: np.ndarray,
                  data_fid_marks: pd.DataFrame, data_extracted: pd.DataFrame) -> np.ndarray:
    """
    This function creates a mask for an image by first identifying fiducial marks
    specific to the given image ID, then extracting text boxes from the image data,
    and finally creating a mask that incorporates these elements.

    Args:
        image_id: A unique identifier for the image.
        image: The image data as a NumPy array.
        data_fid_marks: A DataFrame containing fiducial marks data.
        data_extracted: A DataFrame containing extracted data, including text boxes.

    Returns:
        A NumPy array representing the mask created for the image, incorporating
        fiducial marks and text boxes. 1 means the pixel is unmasked, 0 means the pixel is masked.

    """
    # Get the fid marks for the specific image_id
    fid_marks_row = data_fid_marks.loc[data_fid_marks['image_id'] == image_id].squeeze()

    # Create fid mark dict using dictionary comprehension
    fid_dict = {i: (fid_marks_row[f'fid_mark_{i}_x'], fid_marks_row[f'fid_mark_{i}_y']) for
                i in range(1, 5)}

    # get the text boxes of the image
    text_string = data_extracted.loc[data_extracted['image_id'] == image_id]['text_bbox'].iloc[0]

    # make all text strings to lists
    if len(text_string) > 0:
        if "[" not in text_string:
            text_string = "[" + text_string + "]"

        # create text-boxes list
        text_boxes = [tuple(group) for group in eval(text_string.replace(";", ","))]
    else:
        text_boxes = None

    # load the mask
    mask = cm.create_mask(image, fid_dict, text_boxes, use_default_fiducials=True)

    return mask


def _save_results(georef_type: str, image_id: str, image: np.ndarray,
                  transform: np.ndarray, residuals: np.ndarray,
                  tps: np.ndarray, conf: np.ndarray, month: int):
    """
    This function saves multiple components of the geo-referencing process including the
    geo-referenced image, transformation matrix, and control points with their confidence
    levels. It also generates and saves a footprint of the geo-referenced image in a shapefile,
    incorporating relevant metadata.

    Args:
        georef_type: The type of geo-referencing performed (e.g., 'sat', 'img', or 'calc).
        image_id: A unique identifier for the image.
        image: The image data as a NumPy array.
        transform: The 2D transformation matrix applied to the image.
        residuals: An array of residuals from the transformation process.
        tps: An array of [x, y] points used in the transformation.
        conf: An array of confidence values corresponding to each point in `tps`.
        month: The month in which the image was acquired.
    """

    # save the geo-referenced image
    path_img = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}.tif"
    at.apply_transform(image, transform, save_path=path_img)

    # save the transform
    path_transform = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}_transform.txt"

    # noinspection PyTypeChecker
    np.savetxt(path_transform, transform.reshape(3, 3), fmt='%.5f')

    # save the points, conf and residuals
    path_points = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}_points.txt"
    tps_conf = np.concatenate([tps, conf.reshape(-1, 1), residuals.reshape((-1, 1))], axis=1)

    # noinspection PyTypeChecker
    np.savetxt(path_points, tps_conf, fmt=['%i', '%i', '%.2f', '%.2f', '%.3f', '%.3f'])

    # calculate average values
    # noinspection PyTypeChecker
    conf_mean: float = np.mean(conf)
    # noinspection PyTypeChecker
    residuals_mean: float = np.mean(residuals)

    # define attributes for shapefile
    attributes = {
        'image_id': image_id,
        'month': month,
        'num_tps': tps.shape[0],
        'avg_conf': round(conf_mean, 3),
        'avg_resi': round(residuals_mean, 3),
    }
    attributes = pd.DataFrame.from_dict(attributes, orient='index').T

    # convert the image to a footprint
    footprint = citf.convert_image_to_footprint(image, transform)

    # save footprint to shp file
    path_shp_file = f"{DEFAULT_SAVE_FOLDER}/{georef_type}.shp"
    eg.export_geometry(footprint, path_shp_file,
                       attributes=attributes, key_field="image_id",
                       overwrite_file=False,
                       overwrite_entry=True, attach=True)


def _load_processed_images(filename):
    full_path = os.path.join(DEFAULT_SAVE_FOLDER, filename)
    if os.path.isfile(full_path):
        df = pd.read_csv(full_path, delimiter=";")
        df.set_index('id', inplace=True)
        return df.to_dict(orient='index')
    else:
        return None


if __name__ == "__main__":
    # load all ids from the csv file as pandas dataframe
    csv_path = "/data_1/ATM/data_1/georef/sat_processed_images.csv"
    processed_images = pd.read_csv(csv_path, delimiter=";")

    # remove all images that are already geo-referenced
    processed_images = processed_images[processed_images['status'] != "georeferenced"]

    # remove all images that are not complex enough
    processed_images = processed_images[processed_images['reason'] != "complexity"]

    # get the image ids as a list
    image_ids = processed_images['id'].tolist()

    georef_adapted(image_ids,
                   overwrite_adapted=OVERWRITE_ADAPTED,
                   retry_missing_adapted=RETRY_MISSING_ADAPTED,
                   retry_failed_adapted=RETRY_FAILED_ADAPTED,
                   retry_invalid_adapted=RETRY_INVALID_ADAPTED,
                   min_complexity=MIN_COMPLEXITY,
                   verify_image_positions=VERIFY_IMAGE_POSITIONS,
                   distance_threshold=DISTANCE_THRESHOLD,
                   catch=False)
