import logging
import os.path
import numpy as np
import pandas as pd

from tqdm import tqdm

# import base functions
import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.base.find_overlapping_images as foi
import src.base.modify_csv as mc

# import export functions
import src.export.export_geometry as eg

# import extract function
import src.extract.extract_ids as ei

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

# settings for verifying
DISTANCE_THRESHOLD = 100

# settings for calc
CALC_TYPES = ["sat", "img"]

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
           overwrite_sat=False, overwrite_img=False, overwrite_calc=False,
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

    # set the path to the csv file with processed images
    csv_path = DEFAULT_SAVE_FOLDER + "/processed_images.csv"

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

            # check if image is already invalid
            if (processed_images.get(image_id, {}).get('method') == "sat" and
                    processed_images.get(image_id, {}).get('status') == "invalid" and
                    not retry_invalid_sat):
                print(f"{image_id} already invalid")
                continue

            # check if image did already fail
            if (processed_images.get(image_id, {}).get('method') == "sat" and
                    processed_images.get(image_id, {}).get('status') == "failed" and
                    not retry_failed_sat):
                print(f"{image_id} already failed")
                continue

            # try to geo-reference the image
            try:

                # ignore images with a too low complexity
                if data_extracted.loc[data_extracted['image_id'] == image_id]['complexity'].iloc[0] < min_complexity:
                    processed_images[image_id] = {"method": "sat", "status": "failed",
                                                  "reason": "complexity"}
                    continue

                print(f"Geo-reference {image_id} with satellite")

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
                    _save_results("sat", image_id, image, transform, residuals, tps, conf, month)
                    processed_images[image_id] = {"method": "sat", "status": "georeferenced",
                                                  "reason": ""}

                # set image to invalid if position is wrong
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
                    mc.modify_csv(csv_path, image_id, "add", processed_images[image_id], overwrite=True)

        # now that we have more images, verify the images again to check for wrong positions
        if verify_image_positions:

            # load footprint and ids of images that are already geo-referenced by satellite
            path_sat_shapefile = "/data_1/ATM/data_1/georef/sat.shp"
            sat_shape_data = lsd.load_shape_data(path_sat_shapefile)

            # iterate all images
            for image_id in tqdm(processed_images):

                footprint = sat_shape_data.loc[
                    sat_shape_data['image_id'] == image_id].geometry.iloc[0]
                line_footprints = sat_shape_data.loc[
                    sat_shape_data['image_id'][2:6] == image_id[2:6]].geometry

                valid_image = vip.verify_image_position(footprint, line_footprints, distance_threshold)

                # set image to invalid if position is wrong
                if not valid_image:
                    processed_images[image_id] = {"method": "sat", "status": "invalid",
                                                  "reason": "position"}
                    mc.modify_csv(csv_path, image_id, "add", processed_images[image_id], overwrite=True)

    # geo-reference with image
    if georef_with_image:

        print("Start geo-referencing with image")

        for image_id in tqdm(input_ids):

            # skip images that are already geo-referenced with satellite
            if (processed_images.get(image_id, {}).get('method') == "sat" and
                    processed_images.get(image_id, {}).get('status') == "georeferenced"):
                print(f"{image_id} already geo-referenced with satellite")
                continue

            # check if image is already geo-referenced
            if (processed_images.get(image_id, {}).get('method') == "img" and
                    processed_images.get(image_id, {}).get('status') == "georeferenced" and
                    not overwrite_img):
                print(f"{image_id} already geo-referenced with image")
                continue

            # check if image is already invalid
            if (processed_images.get(image_id, {}).get('method') == "img" and
                    processed_images.get(image_id, {}).get('status') == "invalid" and
                    not retry_invalid_img):
                print(f"{image_id} already invalid")
                continue

            # check if image did already fail
            if (processed_images.get(image_id, {}).get('method') == "img" and
                    processed_images.get(image_id, {}).get('status') == "failed" and
                    not retry_failed_img):
                print(f"{image_id} already failed")
                continue

            # try to geo-reference the image
            try:

                # find overlapping images
                image_ids = foi.find_overlapping_images(image_id)

                # ignore images with no neighbouring images
                georeferenced_images = []
                georeferenced_transforms = []

                print(f"Geo-reference {image_id} with image")

                # load the image
                image = li.load_image(image_id)

                # load the mask
                mask = _prepare_mask(image_id, image, data_fid_marks, data_extracted)

                # check if all required data is there
                if mask is None:
                    processed_images[image_id] = {"method": "img", "status": "missing_data",
                                                  "reason": "mask"}
                    continue

                # the actual geo-referencing
                transform, residuals, tps, conf = georef_img.georeference(image,
                                                                          georeferenced_images,
                                                                          georeferenced_transforms,
                                                                          mask=mask, georeferenced_masks=None)

                # skip images we can't geo-reference
                if transform is None:
                    processed_images[image_id] = {"method": "img", "status": "failed",
                                                  "reason": "no_transform"}
                    continue

                # verify the geo-referenced image
                valid_image, reason = vig.verify_image_geometry(image, transform)

                # save valid images
                if valid_image:

                    # get the month of the image
                    month = data_images.loc[
                        data_images['image_id'] == image_id]['date_month'].iloc[0]

                    _save_results("img", image_id, image, transform, residuals, tps, conf, month)
                    processed_images[image_id] = {"method": "img", "status": "georeferenced",
                                                  "reason": ""}

                # set image to invalid if position is wrong
                else:
                    processed_images[image_id] = {"method": "img", "status": "invalid",
                                                  "reason": reason}

                print(f"Geo-referencing of {image_id} finished")

            # manually catch the keyboard interrupt
            except KeyboardInterrupt:
                keyboard_interrupt = True

            # that means something in the geo-referencing process fails
            except (Exception,) as e:

                processed_images[image_id] = {"method": "img", "status": "failed",
                                              "reason": "exception"}

                if catch is False:
                    raise e

                print(f"Geo-referencing of {image_id} failed")

            # we always want to add information to the csv file
            finally:

                if keyboard_interrupt is False:
                    mc.modify_csv(csv_path, image_id, "add", processed_images[image_id], overwrite=True)

    # geo-reference with calc
    if georef_with_calc:

        print("Start geo-referencing with calc")

        georeferenced_footprints = []
        georeferenced_ids = []

        if "sat" in calc_types:
            path_sat_shapefile = "/data_1/ATM/data_1/georef/sat.shp"
            sat_shape_data = lsd.load_shape_data(path_sat_shapefile)
            sat_shapes = sat_shape_data.geometry
            sat_ids = sat_shape_data['image_id'].tolist()

            georeferenced_footprints.extend(sat_shapes)
            georeferenced_ids.extend(sat_ids)

        if "img" in calc_types:
            path_img_shapefile = "/data_1/ATM/data_1/georef/img.shp"
            img_shape_data = lsd.load_shape_data(path_img_shapefile)
            img_shapes = img_shape_data.geometry
            img_ids = img_shape_data['image_id'].tolist()

            georeferenced_footprints.extend(img_shapes)
            georeferenced_ids.extend(img_ids)

        for image_id in tqdm(input_ids):

            # skip images that are already geo-referenced with satellite or image
            if (processed_images.get(image_id, {}).get('method') in ["sat", "img"] and
                    processed_images.get(image_id, {}).get('status') == "georeferenced"):
                print(f"{image_id} already geo-referenced with satellite or image")
                continue

            # check if image is already geo-referenced
            if (processed_images.get(image_id, {}).get('method') == "calc" and
                    processed_images.get(image_id, {}).get('status') == "georeferenced" and
                    not overwrite_calc):
                print(f"{image_id} already geo-referenced with calc")
                continue

            # check if image is already invalid
            if (processed_images.get(image_id, {}).get('method') == "calc" and
                    processed_images.get(image_id, {}).get('status') == "invalid" and
                    not retry_invalid_calc):
                print(f"{image_id} already invalid")
                continue

            # check if image did already fail
            if (processed_images.get(image_id, {}).get('method') == "calc" and
                    processed_images.get(image_id, {}).get('status') == "failed" and
                    not retry_failed_calc):
                print(f"{image_id} already failed")
                continue

            # try to geo-reference the image
            try:

                print(f"Geo-reference {image_id} with calc")

                # load the image
                image = li.load_image(image_id)

                # the actual geo-referencing
                transform, residuals, tps, conf = georef_calc.georeference(image, image_id,
                                                                           georeferenced_ids,
                                                                           georeferenced_footprints)

                # skip images we can't geo-reference
                if transform is None:
                    processed_images[image_id] = {"method": "calc", "status": "failed",
                                                  "reason": "no_transform"}
                    continue

                # verify the geo-referenced image
                valid_image, reason = vig.verify_image_geometry(image, transform)

                # save valid images
                if valid_image:

                    # get the month of the image
                    month = data_images.loc[
                        data_images['image_id'] == image_id]['date_month'].iloc[0]

                    _save_results("calc", image_id, image, transform, residuals, tps, conf, month)
                    processed_images[image_id] = {"method": "calc", "status": "georeferenced",
                                                  "reason": ""}

                # set image to invalid if position is wrong
                else:
                    processed_images[image_id] = {"method": "calc", "status": "invalid",
                                                  "reason": reason}

                print(f"Geo-referencing of {image_id} finished")

            # manually catch the keyboard interrupt
            except KeyboardInterrupt:
                keyboard_interrupt = True

            # that means something in the geo-referencing process fails
            except (Exception,) as e:

                processed_images[image_id] = {"method": "calc", "status": "failed",
                                              "reason": "exception"}

                if catch is False:
                    raise e

                print(f"Geo-referencing of {image_id} failed")

            # we always want to add information to the csv file
            finally:

                if keyboard_interrupt is False:
                    mc.modify_csv(csv_path, image_id, "add", processed_images[image_id], overwrite=True)


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

    # create text-boxes list
    text_boxes = [tuple(group) for group in eval(text_string.replace(";", ","))]

    # load the mask
    mask = cm.create_mask(image, fid_dict, text_boxes)

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
    np.savetxt(path_transform, transform.reshape(3,3), fmt='%.5f')

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
        _processed_images = _processed_images.to_dict(orient='index')
    else:
        _processed_images = None

    # call the actual geo-referencing function
    georef(_input_ids, _processed_images,
           georef_with_satellite=GEOREF_WITH_SATELLITE,
           georef_with_image=GEOREF_WITH_IMAGE,
           georef_with_calc=GEOREF_WITH_CALC,
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
