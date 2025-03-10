# Library imports
import copy
import glob
import os.path
from xmlrpc.client import Fault

import numpy as np
import pandas as pd
import shutil
import time
import torch.cuda
from datetime import datetime
from tqdm import tqdm

# import base functions
import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.base.find_overlapping_images_id as foi
import src.base.modify_csv as mc

# import export functions
import src.export.export_geometry as eg

# import extract function
import src.other.extract.extract_ids_by_area as ei

# import georef functions
import src.georef.georef_sat as gs
import src.georef.georef_img as gi
import src.georef.georef_calc as gc

# import georef snippet functions
import src.georef.snippets.apply_transform as at
import src.georef.snippets.convert_image_to_footprint as citf
import src.georef.snippets.verify_image_geometry as vig
import src.georef.snippets.verify_image_position as vip

# import load functions
import src.load.load_image as li
import src.load.load_shape_data as lsd
import src.load.load_transform as lt

# define save folder
DEFAULT_SAVE_FOLDER = "/data/ATM/data_1/georef"

# define input bounds or image ids
INPUT_TYPE = "flight_path"  # can be either "bounds" or "ids" or "flight_path" or "all"
BOUNDS = []  # min_x, min_y, max_x, max_y
IMAGE_IDS = []
FLIGHT_PATHS = [1815] # , 2149, 2139, 2142, 2141, 2158, 2140, 1816, 1821]

# which type of geo-referencing should be done
GEOREF_WITH_SATELLITE = False
GEOREF_WITH_IMAGE = False
GEOREF_WITH_CALC = True


auto_reject=["CA164432V0040", "CA164432V0066", "CA212132V0078", "CA212132V0079", "CA181832V0005", "CA207432V0175",
             "CA184332V0002", "CA184332V0008", "CA214232V0229", "CA196832V0141", "CA214032V0430", "CA216432V0236",
             "CA216432V0238", "CA216332V0157", "CA213732V0059", "CA215232V0318", "CA215232V0317"]

# settings for sat
MIN_COMPLEXITY = 0.05
MIN_TPS_SAT = 20
VERIFY_IMAGE_POSITIONS = False
VERIFY_IMAGE_GEOMETRY_CALC=False

# settings for verifying
DISTANCE_THRESHOLD = 100  # TODO UPDATE THIS!

# settings for calc
CALC_TYPES = ["sat", "img"]

# define if images of a certain type should be overwritten
OVERWRITE_SAT = False
OVERWRITE_IMG = True
OVERWRITE_CALC = True

# define if images with missing data should be done again
RETRY_MISSING_SAT = True

# define if failed images should be done again
RETRY_FAILED_SAT = True
RETRY_FAILED_IMG = True
RETRY_FAILED_CALC = True

# define if invalid images should be done again
RETRY_INVALID_SAT = True
RETRY_INVALID_IMG = True
RETRY_INVALID_CALC = True


def georef(input_ids, processed_images_sat=None, processed_images_adapted=None,
           processed_images_img=None, processed_images_calc=None,
           georef_with_satellite=True, georef_with_image=True, georef_with_calc=True,
           overwrite_sat=False, overwrite_img=False, overwrite_calc=False,
           retry_missing_sat=True,
           retry_failed_sat=True, retry_failed_img=True, retry_failed_calc=True,
           retry_invalid_sat=True, retry_invalid_img=True, retry_invalid_calc=True,
           min_complexity=0.0, verify_image_positions=True, distance_threshold=100,
           calc_types=None,
           catch=False):
    print("Start georef")

    # set the right calc types
    if calc_types is None:
        calc_types = ["sat", "img"]

    # flag required for the loop
    keyboard_interrupt = False

    # init the geo-reference objects
    georef_sat = gs.GeorefSatellite(min_tps_final=MIN_TPS_SAT, enhance_image=False,
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

    # set the path to the csv files with processed images
    csv_path_sat = DEFAULT_SAVE_FOLDER + "/sat_processed_images.csv"
    csv_path_img = DEFAULT_SAVE_FOLDER + "/img_processed_images.csv"
    csv_path_calc = DEFAULT_SAVE_FOLDER + "/calc_processed_images.csv"

    # init processed images if not given
    if processed_images_sat is None:
        processed_images_sat = {}
    if processed_images_adapted is None:
        processed_images_adapted = {}
    if processed_images_img is None:
        processed_images_img = {}
    if processed_images_calc is None:
        processed_images_calc = {}

    # geo-reference with satellite
    if georef_with_satellite:

        print("Start geo-referencing with satellite")

        # iterate all images
        for img_counter, image_id in enumerate(tqdm(input_ids)):

            # check if image is already geo-referenced with satellite
            if (processed_images_sat.get(image_id, {}).get('status') == "georeferenced" and
                    not overwrite_sat):
                print(f"{image_id} already geo-referenced")
                continue

            # check if image is already geo-referenced with satellite adapted
            if (processed_images_adapted.get(image_id, {}).get('status') == "georeferenced" and
                    not overwrite_sat):
                print(f"{image_id} already geo-referenced")
                continue

            # check if image is already invalid
            if (processed_images_sat.get(image_id, {}).get('status') == "invalid" and
                    not retry_invalid_sat):
                print(f"{image_id} already invalid")
                continue

            # check if image did already fail
            if (processed_images_sat.get(image_id, {}).get('status') == "failed" and
                    not retry_failed_sat):
                print(f"{image_id} already failed")
                continue

            # check if image was missing data
            if (processed_images_sat.get(image_id, {}).get('status') == "missing_data" and
                    not retry_missing_sat):
                print(f"{image_id} already missing data")
                continue

            # backup the processed_images every 10th iteration
            if img_counter % 10 == 0:
                source_directory = os.path.dirname(csv_path_sat)
                backup_directory = os.path.join(source_directory, 'backup')
                filename = os.path.basename(csv_path_sat)
                backup_path = os.path.join(backup_directory, filename)
                shutil.copy(csv_path_sat, backup_path)

            # start the timer
            start = time.time()

            # try to geo-reference the image
            try:

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                # ignore images with a too low complexity
                if data_extracted.loc[data_extracted['image_id'] == image_id]['complexity'].iloc[0] < min_complexity:
                    processed_images_sat[image_id] = {"method": "sat", "status": "failed",
                                                      "reason": "complexity", "date": date_time_str}
                    print(f"{image_id} has low complexity")
                    continue

                # load the image
                image = li.load_image(image_id, catch=True)

                if image is None:
                    # get datetime
                    now = datetime.now()
                    date_time_str = now.strftime("%d.%m.%Y %H:%M")

                    processed_images_sat[image_id] = {"method": "sat", "status": "failed",
                                                      "reason": "image", "date": date_time_str}
                    print(f"{image_id} could not be loaded")
                    continue

                # load the mask
                try:
                    mask = _prepare_mask(image_id, image, data_fid_marks, data_extracted)
                except (Exception,):
                    mask = None

                # get the approx footprint of the image
                approx_footprint = data_extracted.loc[
                    data_extracted['image_id'] == image_id]['footprint_approx'].iloc[0]

                # get the azimuth of the image
                azimuth = data_images.loc[
                    data_images['image_id'] == image_id]['azimuth'].iloc[0]

                # get the month of the image
                month = data_images.loc[
                    data_images['image_id'] == image_id]['date_month'].iloc[0]

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                # check if all required data is there
                if approx_footprint is None:
                    processed_images_sat[image_id] = {"method": "sat", "status": "missing_data",
                                                      "reason": "approx_footprint", "time": "",
                                                      "date": date_time_str}
                    print(f"{image_id} has no approx_footprint")
                    continue
                elif mask is None:
                    processed_images_sat[image_id] = {"method": "sat", "status": "missing_data",
                                                      "reason": "mask", "time": "",
                                                      "date": date_time_str}
                    print(f"{image_id} has no mask")
                    continue
                elif azimuth is None:
                    processed_images_sat[image_id] = {"method": "sat", "status": "missing_data",
                                                      "reason": "azimuth", "time": "",
                                                      "date": date_time_str}
                    print(f"{image_id} has no azimuth")
                    continue

                print(f"Geo-reference {image_id} with satellite")

                # we need to adapt the azimuth to account for EPSG:3031
                azimuth = 360 - azimuth + 90

                # the actual geo-referencing
                transform, residuals, tps, conf = georef_sat.georeference(image, approx_footprint,
                                                                          mask, azimuth, month)

                # skip images we can't geo-reference
                if transform is None:

                    georef_time = round(time.time() - start)

                    # get datetime
                    now = datetime.now()
                    date_time_str = now.strftime("%d.%m.%Y %H:%M")

                    # images failed due to exception
                    if tps is None:
                        processed_images_sat[image_id] = {"method": "sat", "status": "failed",
                                                          "reason": "exception", "time": georef_time,
                                                          "date": date_time_str}
                    # too few tps
                    else:
                        processed_images_sat[image_id] = {"method": "sat", "status": "failed",
                                                          "reason": f"too_few_tps:{tps.shape[0]}",
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
                    processed_images_sat[image_id] = {"method": "sat", "status": "georeferenced",
                                                      "reason": "", "time": georef_time,
                                                      "date": date_time_str}

                # set image to invalid if position is wrong
                else:
                    processed_images_sat[image_id] = {"method": "sat", "status": "invalid",
                                                      "reason": reason, "time": georef_time,
                                                      "date": date_time_str}

                print(f"Geo-referencing of {image_id} finished")

            # manually catch the keyboard interrupt
            except KeyboardInterrupt:
                keyboard_interrupt = True

            # that means something in the geo-referencing process fails
            except (Exception,) as e:

                fail_time = round(time.time() - start)

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                processed_images_sat[image_id] = {"method": "sat", "status": "failed",
                                                  "reason": "exception",
                                                  "time": fail_time,
                                                  "date": date_time_str}

                if catch is False:
                    raise e

                print(f"Geo-referencing of {image_id} failed")

            # we always want to add information to the csv file
            finally:

                if keyboard_interrupt is False:
                    mc.modify_csv(csv_path_sat, image_id, "add", processed_images_sat[image_id], overwrite=True)

                # clear the memory
                torch.cuda.empty_cache()

    # now that we have more images, verify the images again to check for wrong positions
    if verify_image_positions:

        # load footprint and ids of images that are already geo-referenced by satellite
        path_sat_shapefile = "/data/ATM/data_1/georef/footprints/sat_footprints.shp"
        sat_shape_data = lsd.load_shape_data(path_sat_shapefile)

        # iterate all images
        for image_id in tqdm(input_ids):

            # check if the id is in sat_shape_data
            if image_id not in sat_shape_data['image_id'].tolist():
                continue

            # get number of line footprints
            num_line_footprints = sat_shape_data.loc[sat_shape_data['image_id'].str[2:6] ==
                                                     image_id[2:6]].shape[0]
            if num_line_footprints < 2:
                continue

            footprint = sat_shape_data.loc[
                sat_shape_data['image_id'] == image_id].geometry.iloc[0]
            line_footprints = sat_shape_data.loc[
                sat_shape_data['image_id'].str[2:6] == image_id[2:6]].geometry

            rejected = vip.verify_image_position(footprint, line_footprints, distance_threshold)

            # set image to invalid if position is wrong
            if rejected:

                print("Rejected image", image_id)

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                # TODO FIX TIMER
                processed_images_sat[image_id] = {"method": "sat", "status": "invalid",
                                                  "reason": "position",
                                                  "date": date_time_str}
                mc.modify_csv(csv_path_sat, image_id, "add", processed_images_sat[image_id], overwrite=True)

    # geo-reference with image
    if georef_with_image:

        print("Start geo-referencing with image")

        sql_string = "SELECT image_ID FROM images_extracted WHERE footprint_type='sat'"
        data = ctd.execute_sql(sql_string, conn)
        sat_tif_files = data['image_id'].tolist()

        # get all fid marks from images
        sql_string_fid_marks = f"SELECT * FROM images_fid_points"
        data_fid_marks = ctd.execute_sql(sql_string_fid_marks, conn)

        # get all text data
        sql_string_extracted = f"SELECT * FROM images_extracted"
        data_extracted = ctd.execute_sql(sql_string_extracted, conn)

        for img_counter, image_id in enumerate(tqdm(input_ids)):

            image_id = str(image_id)

            # skip images that are already geo-referenced with satellite
            if processed_images_sat.get(image_id, {}).get('status') == "georeferenced":
                print(f"{image_id} already geo-referenced with satellite")
                continue

            # skip images that are already geo-referenced with satellite adapted
            if processed_images_adapted.get(image_id, {}).get('status') == "georeferenced":
                print(f"{image_id} already geo-referenced with satellite adapted")
                continue

            # check if image is already geo-referenced
            if (processed_images_img.get(image_id, {}).get('status') == "georeferenced" and
                    not overwrite_img):
                print(f"{image_id} already geo-referenced with image")
                continue

            # check if image is already invalid
            if (processed_images_img.get(image_id, {}).get('status') == "invalid" and
                    not retry_invalid_img):
                print(f"{image_id} already invalid")
                continue

            # check if image did already fail
            if (processed_images_img.get(image_id, {}).get('status') == "failed" and
                    not retry_failed_img):
                print(f"{image_id} already failed")
                continue

            # backup the processed_images every 10th iteration
            if img_counter % 10 == 0:
                source_directory = os.path.dirname(csv_path_img)
                backup_directory = os.path.join(source_directory, 'backup')
                filename = os.path.basename(csv_path_img)
                backup_path = os.path.join(backup_directory, filename)
                shutil.copy(csv_path_img, backup_path)

            # start the timer
            start = time.time()

            # try to geo-reference the image
            try:

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                # create local copy for image_ids and append the current image_id
                image_ids = copy.deepcopy(sat_tif_files)
                image_ids.append(image_id)

                # get overlapping images
                overlap_dict = foi.find_overlapping_images_id(image_ids,
                                                              important_id=image_id, max_id_range=1)
                georeferenced_ids = overlap_dict[image_id]

                if len(georeferenced_ids) == 0:
                    processed_images_img[image_id] = {"method": "img", "status": "failed",
                                                      "reason": "no_neighbours", "time": "",
                                                      "date": date_time_str}
                    print(f"No georeferenced images for {image_id}")
                    continue

                # init lists for georeferenced images
                georeferenced_images = []
                georeferenced_transforms = []
                georeferenced_masks = []

                # set path to transform folder
                transform_folder = "/data/ATM/data_1/georef/transforms"

                # load the georeferenced images
                for georef_id in georeferenced_ids:
                    # load image
                    georef_image = li.load_image(georef_id)
                    georeferenced_images.append(georef_image)

                    # load transform
                    georef_transform = lt.load_transform(transform_folder + "/sat/" +
                                                         georef_id + "_transform.txt", delimiter=" ")
                    georeferenced_transforms.append(georef_transform)

                    # get data for mask
                    georeferenced_fid_marks = data_fid_marks[data_fid_marks['image_id'] == image_id]
                    georeferenced_extracted = data_extracted[data_extracted['image_id'] == image_id]

                    # load mask
                    georef_mask = _prepare_mask(image_id, georef_image,
                                                georeferenced_fid_marks, georeferenced_extracted)
                    georeferenced_masks.append(georef_mask)

                print(f"Geo-reference {image_id} with image")

                # load the image
                image = li.load_image(image_id, catch=True)

                if image is None:
                    # get datetime
                    now = datetime.now()
                    date_time_str = now.strftime("%d.%m.%Y %H:%M")

                    processed_images_img[image_id] = {"method": "img", "status": "failed",
                                                      "reason": "image", "time": "",
                                                      "date": date_time_str}
                    continue

                # load the mask
                mask = _prepare_mask(image_id, image, data_fid_marks, data_extracted)

                # check if all required data is there
                if mask is None:
                    # get datetime
                    now = datetime.now()
                    date_time_str = now.strftime("%d.%m.%Y %H:%M")

                    processed_images_img[image_id] = {"method": "img", "status": "missing_data",
                                                      "reason": "mask", "time": "",
                                                      "date": date_time_str}
                    continue

                # the actual geo-referencing
                transform, residuals, tps, conf = georef_img.georeference(image,
                                                                          georeferenced_images,
                                                                          georeferenced_transforms,
                                                                          mask=mask,
                                                                          georeferenced_masks=georeferenced_masks)

                # skip images we can't geo-reference
                if transform is None:
                    georef_time = round(time.time() - start)

                    # get datetime
                    now = datetime.now()
                    date_time_str = now.strftime("%d.%m.%Y %H:%M")

                    processed_images_img[image_id] = {"method": "img", "status": "failed",
                                                      "reason": "no_transform",
                                                      "time": georef_time,
                                                      "date": date_time_str}
                    continue

                # verify the geo-referenced image
                valid_image, reason = vig.verify_image_geometry(image, transform)

                georef_time = round(time.time() - start)

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                print(image_id, valid_image)

                # save valid images
                if valid_image:

                    # get the month of the image
                    month = data_images.loc[
                        data_images['image_id'] == image_id]['date_month'].iloc[0]

                    _save_results("img", image_id, image, transform, residuals, tps, conf, month)
                    processed_images_img[image_id] = {"method": "img", "status": "georeferenced",
                                                      "reason": "", "time": georef_time,
                                                      "date": date_time_str}

                # set image to invalid if position is wrong
                else:
                    processed_images_img[image_id] = {"method": "img", "status": "invalid",
                                                      "reason": reason, "time": georef_time,
                                                      "date": date_time_str}

                print(f"Geo-referencing of {image_id} finished")

            # manually catch the keyboard interrupt
            except KeyboardInterrupt:
                keyboard_interrupt = True

            # that means something in the geo-referencing process fails
            except (Exception,) as e:

                fail_time = round(time.time() - start)

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                processed_images_img[image_id] = {"method": "img", "status": "failed",
                                                  "reason": "exception",
                                                  "time": fail_time,
                                                  "date": date_time_str}

                if catch is False:
                    raise e

                print(f"Geo-referencing of {image_id} failed")

            # we always want to add information to the csv file
            finally:

                if keyboard_interrupt is False:
                    mc.modify_csv(csv_path_img, image_id, "add", processed_images_img[image_id], overwrite=True)

                # clear the memory
                torch.cuda.empty_cache()

    # geo-reference with calc
    if georef_with_calc:

        print("Start geo-referencing with calc")

        georeferenced_footprints = []
        georeferenced_ids = []

        if "sat" in calc_types:

            # load shapes
            path_sat_shapefile = "/data/ATM/data_1/georef/footprints/sat_footprints.shp"
            sat_shape_data = lsd.load_shape_data(path_sat_shapefile)
            sat_shapes = sat_shape_data.geometry
            sat_ids = sat_shape_data['image_id'].tolist()

            # load csv file
            path_sat_csv = "/data/ATM/data_1/georef/sat_processed_images.csv"
            sat_csv_data = pd.read_csv(path_sat_csv, delimiter=";")

            # Filter sat_csv_data to only include valid images
            if verify_image_positions:
                valid_sat_csv_data = sat_csv_data[sat_csv_data['status'] != 'invalid']
            else:
                valid_sat_csv_data = sat_csv_data[
                    (sat_csv_data['status'] != 'invalid') | (sat_csv_data['reason'] == 'position')]
            valid_sat_ids = valid_sat_csv_data['id'].tolist()

            # Filter sat_shapes and sat_ids to only include valid_sat_ids
            sat_shapes = sat_shapes[sat_shape_data['image_id'].isin(valid_sat_ids)]
            sat_ids = [sat_id for sat_id in sat_ids if sat_id in valid_sat_ids]

            georeferenced_footprints.extend(sat_shapes)
            georeferenced_ids.extend(sat_ids)

        if "img" in calc_types:
            path_img_shapefile = "/data/ATM/data_1/georef/footprints/img_footprints.shp"
            img_shape_data = lsd.load_shape_data(path_img_shapefile)

            # Filter img_shapes and img_ids to only include IDs not already in georeferenced_ids
            filtered_img_data = img_shape_data[~img_shape_data['image_id'].isin(georeferenced_ids)]

            # Extend only with new IDs and shapes
            filtered_img_shapes = filtered_img_data.geometry.tolist()
            filtered_img_ids = filtered_img_data['image_id'].tolist()

            georeferenced_footprints.extend(filtered_img_shapes)
            georeferenced_ids.extend(filtered_img_ids)

        # Verify image positions
        if verify_image_positions:

            final_ids = []
            final_footprints = []

            # load footprint and ids of images that are already geo-referenced by satellite
            path_sat_shapefile = "/data/ATM/data_1/georef/footprints/sat_footprints.shp"
            shape_data = lsd.load_shape_data(path_sat_shapefile)
            # load footprint and ids of images that are already geo-referenced by iamge
            if "img" in calc_types:
                path_img_shapefile = "/data/ATM/data_1/georef/footprints/img_footprints.shp"
                img_shape_data = lsd.load_shape_data(path_img_shapefile)

                # Merge the shapefiles, prioritizing satellite data
                merged_shape_data = pd.concat([shape_data, img_shape_data], ignore_index=True)
                shape_data = merged_shape_data.drop_duplicates(subset='image_id', keep='first')

            if FLIGHT_PATHS is not None:
                str_fps = [str(fp) for fp in FLIGHT_PATHS]
                shape_data = shape_data[shape_data['image_id'].str[2:6].isin(str_fps)]

            for image_id in georeferenced_ids:

                # Check based on calc_types
                if image_id not in shape_data['image_id'].tolist():
                    continue

                # get number of line footprints
                num_line_footprints = shape_data.loc[shape_data['image_id'].str[2:6] ==
                                                         image_id[2:6]].shape[0]
                if num_line_footprints < 2:
                    continue

                footprint = shape_data.loc[
                    shape_data['image_id'] == image_id].geometry.iloc[0]
                line_footprints = shape_data.loc[
                    shape_data['image_id'].str[2:6] == image_id[2:6]].geometry



                rejected = vip.verify_image_position(footprint, line_footprints,
                                                     distance_threshold, flight_path=image_id[2:6])

                if rejected is False:
                    final_ids.append(image_id)
                    final_footprints.append(footprint)
                else:
                    print(f"{image_id} rejected")

            georeferenced_ids = final_ids
            georeferenced_footprints = final_footprints

        if verify_image_positions:

            final_ids = []
            final_footprints = []

            # load footprint and ids of images that are already geo-referenced by satellite
            path_sat_shapefile = "/data/ATM/data_1/georef/footprints/sat_footprints.shp"
            shape_data = lsd.load_shape_data(path_sat_shapefile)
            # load footprint and ids of images that are already geo-referenced by iamge
            if "img" in calc_types:
                path_img_shapefile = "/data/ATM/data_1/georef/footprints/img_footprints.shp"
                img_shape_data = lsd.load_shape_data(path_img_shapefile)

                # Merge the shapefiles, prioritizing satellite data
                merged_shape_data = pd.concat([shape_data, img_shape_data], ignore_index=True)
                shape_data = merged_shape_data.drop_duplicates(subset='image_id', keep='first')

            if FLIGHT_PATHS is not None:
                str_fps = [str(fp) for fp in FLIGHT_PATHS]
                shape_data = shape_data[shape_data['image_id'].str[2:6].isin(str_fps)]

            for image_id in georeferenced_ids:

                # Check based on calc_types
                if image_id not in shape_data['image_id'].tolist():
                    continue

                # get number of line footprints
                num_line_footprints = shape_data.loc[shape_data['image_id'].str[2:6] ==
                                                         image_id[2:6]].shape[0]
                if num_line_footprints < 2:
                    continue

                footprint = shape_data.loc[
                    shape_data['image_id'] == image_id].geometry.iloc[0]
                line_footprints = shape_data.loc[
                    shape_data['image_id'].str[2:6] == image_id[2:6]].geometry



                rejected = vip.verify_image_position(footprint, line_footprints,
                                                     distance_threshold, flight_path=image_id[2:6])

                if rejected is False:
                    final_ids.append(image_id)
                    final_footprints.append(footprint)
                else:
                    print(f"{image_id} rejected")

            georeferenced_ids = final_ids
            georeferenced_footprints = final_footprints

        # remove all ids and footprints that are in auto_reject
        filtered_pairs = [(image_id, footprint) for image_id, footprint in zip(georeferenced_ids, georeferenced_footprints) if image_id not in auto_reject]

        if len(filtered_pairs) == 0:
            print("All images were rejected")
            return
        else:
            georeferenced_ids, georeferenced_footprints = map(list, zip(*filtered_pairs))

        # create a list of image_ids that do not have enough images to calculate the transform
        flight_path_not_enough_images = []

        for img_counter, image_id in enumerate(tqdm(input_ids)):

            # skip images that are already geo-referenced with satellite
            if processed_images_sat.get(image_id, {}).get('status') == "georeferenced":
                print(f"{image_id} already geo-referenced with satellite")
                continue

            # skip images that are already geo-referenced with satellite adapted
            if processed_images_adapted.get(image_id, {}).get('status') == "georeferenced":
                print(f"{image_id} already geo-referenced with satellite adapted")
                continue

            # skip images that are already geo-referenced with image
            if processed_images_img.get(image_id, {}).get('status') == "georeferenced":
                print(f"{image_id} already geo-referenced with image")
                continue

            # check if image is already geo-referenced
            if (processed_images_calc.get(image_id, {}).get('status') == "georeferenced" and
                    not overwrite_calc):
                print(f"{image_id} already geo-referenced with calc")
                continue

            # check if image is already invalid
            if (processed_images_calc.get(image_id, {}).get('status') == "invalid" and
                    not retry_invalid_calc):
                print(f"{image_id} already invalid")
                continue

            # check if image did already fail
            if (processed_images_calc.get(image_id, {}).get('status') == "failed" and
                    not retry_failed_calc):
                print(f"{image_id} already failed")
                continue

            # get flight path from image
            flight_path = image_id[2:6]
            if flight_path in flight_path_not_enough_images:
                print("Not enough images to calculate a position")
                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                processed_images_calc[image_id] = {"method": "calc", "status": "failed",
                                                   "reason": "no_transform", "time": "",
                                                   "date": date_time_str}
                continue

            # backup the processed_images every 10th iteration
            if img_counter % 10 == 0:
                source_directory = os.path.dirname(csv_path_calc)
                backup_directory = os.path.join(source_directory, 'backup')
                filename = os.path.basename(csv_path_calc)
                backup_path = os.path.join(backup_directory, filename)
                shutil.copy(csv_path_calc, backup_path)

            # start the timer
            start = time.time()

            # try to geo-reference the image
            try:

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                print(f"Geo-reference {image_id} with calc")

                # load the image
                try:
                    image = li.load_image(image_id)

                    # the actual geo-referencing
                    transform, residuals, tps, conf = georef_calc.georeference(image, image_id,
                                                                               georeferenced_ids,
                                                                               georeferenced_footprints)
                except Exception as e:
                    transform = None

                if type(transform) == str and transform == 'not_enough':
                    # get flight path from image
                    flight_path = image_id[2:6]

                    # add flight path to list
                    flight_path_not_enough_images.append(flight_path)

                    transform = None

                # skip images we can't geo-reference
                if transform is None:
                    # get datetime
                    now = datetime.now()
                    date_time_str = now.strftime("%d.%m.%Y %H:%M")

                    processed_images_calc[image_id] = {"method": "calc", "status": "failed",
                                                       "reason": "no_transform", "time": "",
                                                       "date": date_time_str}
                    continue

                # verify the geo-referenced image
                if VERIFY_IMAGE_GEOMETRY_CALC:
                    valid_image, reason = vig.verify_image_geometry(image, transform, min_pixel_size = 0.05)
                else:
                    valid_image = True
                    reason = ""

                georef_time = round(time.time() - start)

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                # save valid images
                if valid_image:

                    # get the month of the image
                    month = data_images.loc[
                        data_images['image_id'] == image_id]['date_month'].iloc[0]

                    _save_results("calc", image_id, image, transform, residuals, tps, conf, month)
                    processed_images_calc[image_id] = {"method": "calc", "status": "georeferenced",
                                                       "reason": "", "time": georef_time,
                                                       "date": date_time_str}

                # set image to invalid if position is wrong
                else:
                    processed_images_calc[image_id] = {"method": "calc", "status": "invalid",
                                                       "reason": reason, "time": georef_time,
                                                       "date": date_time_str}

                print(f"Geo-referencing of {image_id} finished")

            # manually catch the keyboard interrupt
            except KeyboardInterrupt:
                keyboard_interrupt = True

            # that means something in the geo-referencing process fails
            except (Exception,) as e:

                fail_time = round(time.time() - start)

                # get datetime
                now = datetime.now()
                date_time_str = now.strftime("%d.%m.%Y %H:%M")

                processed_images_calc[image_id] = {"method": "calc", "status": "failed",
                                                   "reason": "exception", "time": fail_time,
                                                   "date": date_time_str}

                if catch is False:
                    raise e

                print(f"Geo-referencing of {image_id} failed")

            # we always want to add information to the csv file
            finally:

                if keyboard_interrupt is False:
                    mc.modify_csv(csv_path_calc, image_id, "add", processed_images_calc[image_id], overwrite=True)


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
    if len(text_string) > 0 and "[" not in text_string:
        text_string = "[" + text_string + "]"

    # create text-boxes list
    text_boxes = [tuple(group) for group in eval(text_string.replace(";", ","))]

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
    path_img = f"{DEFAULT_SAVE_FOLDER}/images/{georef_type}/{image_id}.tif"
    at.apply_transform(image, transform, save_path=path_img)

    # save the transform
    path_transform = f"{DEFAULT_SAVE_FOLDER}/images/{georef_type}/{image_id}_transform.txt"

    # noinspection PyTypeChecker
    print(path_transform)
    np.savetxt(path_transform, transform.reshape(3, 3), fmt='%.5f')

    # save the points, conf and residuals
    path_points = f"{DEFAULT_SAVE_FOLDER}/images/{georef_type}/{image_id}_points.txt"
    tps_conf = np.concatenate([tps, conf.reshape(-1, 1), residuals.reshape((-1, 1))], axis=1)
    # noinspection PyTypeChecker
    print(path_points)
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
    path_shp_file = f"{DEFAULT_SAVE_FOLDER}/footprints/{georef_type}_footprints.shp"

    eg.export_geometry(footprint, path_shp_file,
                       attributes=attributes, key_field="image_id",
                       overwrite_file=False,
                       overwrite_entry=True, attach=True)


if __name__ == "__main__":

    # load image ids with bounds
    if INPUT_TYPE == "bounds":
        # load all approximate positions of images
        path_approx_shape = "/data/ATM/data_1/shapefiles/TMA_Photocenters/TMA_pts_20100927.shp"
        image_positions = lsd.load_shape_data(path_approx_shape)

        # filter the ids inside the bounds
        _input_ids = ei.extract_ids_by_area(BOUNDS, image_positions,
                                            image_directions=["V"], complete_flightpaths=True)

    # image ids are provided
    elif INPUT_TYPE == "ids":
        _input_ids = IMAGE_IDS

    # we just use all ids
    elif INPUT_TYPE == "all":
        _conn = ctd.establish_connection()
        sql_string = "SELECT image_id FROM images"
        _input_ids = ctd.execute_sql(sql_string, _conn).image_id.tolist()

    elif INPUT_TYPE == "flight_path":

        if isinstance(FLIGHT_PATHS, str):
            FLIGHT_PATHS = [FLIGHT_PATHS]

        flight_paths_str = ", ".join([f"'{fp}'" for fp in FLIGHT_PATHS])

        _conn = ctd.establish_connection()
        sql_string = f"SELECT image_id FROM images WHERE SUBSTRING(image_id, 3, 4) IN ({flight_paths_str})"
        _input_ids = ctd.execute_sql(sql_string, _conn).image_id.tolist()
    else:
        raise ValueError("INPUT_TYPE must be either 'bounds', 'flight_path', 'ids' or 'all'")

    # filter image ids for only vertical images
    _input_ids = [img_id for img_id in _input_ids if "V" in img_id]

    # check if there is a status csv and load it
    def _load_processed_images(filename):
        full_path = os.path.join(DEFAULT_SAVE_FOLDER, filename)
        if os.path.isfile(full_path):
            df = pd.read_csv(full_path, delimiter=";")
            df.set_index('id', inplace=True)
            return df.to_dict(orient='index')
        else:
            return None

    if len(_input_ids) == 0:
        raise ValueError("No images found")

    _processed_images_sat = _load_processed_images("sat_processed_images.csv")
    _processed_images_adapted = _load_processed_images("adapted_processed_images.csv")
    _processed_images_img = _load_processed_images("img_processed_images.csv")
    _processed_images_calc = _load_processed_images("calc_processed_images.csv")

    print(f"Georef {_input_ids}")

    # call the actual geo-referencing function
    georef(_input_ids,
           processed_images_sat=_processed_images_sat,
           processed_images_adapted=_processed_images_adapted,
           processed_images_img=_processed_images_img,
           processed_images_calc=_processed_images_calc,
           georef_with_satellite=GEOREF_WITH_SATELLITE,
           georef_with_image=GEOREF_WITH_IMAGE,
           georef_with_calc=GEOREF_WITH_CALC,
           overwrite_sat=OVERWRITE_SAT,
           overwrite_img=OVERWRITE_IMG,
           overwrite_calc=OVERWRITE_CALC,
           retry_missing_sat=RETRY_MISSING_SAT,
           retry_failed_sat=RETRY_FAILED_SAT,
           retry_failed_img=RETRY_FAILED_IMG,
           retry_failed_calc=RETRY_FAILED_CALC,
           retry_invalid_sat=RETRY_INVALID_SAT,
           retry_invalid_img=RETRY_INVALID_IMG,
           retry_invalid_calc=RETRY_INVALID_CALC,
           min_complexity=MIN_COMPLEXITY,
           verify_image_positions=VERIFY_IMAGE_POSITIONS,
           distance_threshold=DISTANCE_THRESHOLD,
           calc_types=CALC_TYPES,
           catch=False)
