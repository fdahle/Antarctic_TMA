import logging

import numpy as np
from tqdm import tqdm

# import base functions
import src.base.connect_to_database as ctd
import src.base.create_mask as cm

# import export functions
import src.export.export_geometry as eg

# import load functions
import src.load.load_image as li

# import georef functions
import src.georef.georef_sat as gs
import src.georef.georef_img as gi
import src.georef.georef_calc as gc

# import georef snippet functions
import src.georef.snippets.apply_transform as at
import src.georef.snippets.convert_image_to_footprint as citf
import src.georef.snippets.verify_image_geometry as vi

DEFAULT_SAVE_FOLDER = ""

def georef(georef_with_satellite=True, georef_with_image=True, georef_with_calc=True):

    # init the geo-reference objects
    georefSat = gs.GeorefSatellite()
    georefImg = gi.GeorefImage()
    georefCalc = gc.GeorefCalc()

    # establish connection to the database
    conn = ctd.establish_connection()

    # get month and angles from the database
    sql_string_images = "SELECT image_id, azimuth, date_month FROM images"
    data_images = ctd.execute_sql(sql_string_images, conn)

    # get fid mark data
    sql_string_fid_marks = "SELECT * FROM images_fid_points"
    data_fid_marks = ctd.execute_sql(sql_string_fid_marks)

    # get extracted data
    sql_string_extracted = "SELECT image_id, text_bbox, ST_AsText(footprint_approx) AS footprint_approx " \
                           "FROM images_extracted"
    data_extracted = ctd.execute_sql(sql_string_extracted, conn)

    # here we save during runtime the images and their status
    processed_images = {}

    # geo-reference with satellite
    for image_id in tqdm(input_ids):

        try:
            # load the image
            image = li.load_image(image_id)

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
            transform, residuals, tps, conf = georefSat.georeference(image, approx_footprint,
                                                                     mask, azimuth, month)

            # verify the geo-referenced image
            valid_image, reason = vi.verify_image(image, transform)

            if valid_image:
               _save_results(image_id, image, transform, residuals, tps, conf)
               processed_images[image_id] = {"status": "georeferenced", "reason": ""}
            else:
               processed_images[image_id] = {"status": "invalid", "reason": "failed"}

        # that means the geo-referencing fails
        except:
            # TODO
            pass

    # now that we have more images, verify the images again to check for wrong positions
    for image_id in processed_images:

        # load the image
        image = li.load_image(image_id)

        vi.verify_image()

    # geo-reference with image
    for image_id in tqdm(input_ids):

        # load the image
        image = li.load_image(image_id)

        # load the mask
        mask = cm.create_mask(image, fid_marks, text_boxes)


        georefImg.georeference(image, mask)

    # geo-reference with calc
    for image_id in tqdm(input_ids):
        pass

def _create_mask():
    pass

def _save_results(georef_type, image_id, image, transform, residuals, tps, conf):

    path_img = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}.tif"
    path_transform = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}_transform.txt"
    path_points = f"{DEFAULT_SAVE_FOLDER}/{georef_type}/{image_id}_points.txt"

    path_shp_file = f"{DEFAULT_SAVE_FOLDER}/sat.shp"

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

    input_ids = get_ids()

    georef(input_ids)