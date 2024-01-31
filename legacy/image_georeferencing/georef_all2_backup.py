import copy
import geopandas as gpd
import glob
import numpy as np
import os
import pandas as pd
import shapely.wkt
import shutil
import time

from datetime import datetime
from shapely.geometry import Point, LineString
from time import sleep
from tqdm import tqdm

import base.connect_to_db as ctd
import base.modify_csv as mc
import base.modify_shape as ms
import base.print_v as p

import image_georeferencing.georef_sat as gs
import image_georeferencing.georef_img as gi
import image_georeferencing.georef_calc as gc

import image_georeferencing.sub.buffer_footprint as bf
import image_georeferencing.sub.calc_camera_position as ccp
import image_georeferencing.sub.calc_footprint_approx as cfa
import image_georeferencing.sub.check_georef_image as cgi
import image_georeferencing.sub.derive_image_position as dip

# enable/disable certain parts of georef_all
step_calc_approx_footprint = False
step_georef_with_satellite_basic = True
step_georef_with_satellite_estimated = True
step_georef_with_image = False
step_georef_with_calc = False

# overwriting settings
overwrite_approx_footprint = False
overwrite_satellite_footprint = False
overwrite_image_footprint = False
overwrite_calc_footprint = False

# catch settings
catch_approx_footprint = True

retry_too_few_tps_sat = True  # should we retry images that had too few tps before? (for sat loop)
retry_too_few_tps_sat_est = True  # should we retry images that had too few tps before? (for sat est loop)
retry_too_few_tps_img = False  # should we retry images that had too few tps before? (for img loop)
retry_too_few_tps_calc = False  # should we retry images that had too few tps before? (for calc loop)
retry_failed_sat = True  # should we retry images that failed before? (for sat loop)
retry_failed_sat_est = True  # should we retry images that failed before? (for sat est loop)
retry_failed_img = False  # should we retry images that failed before? (for img loop)
retry_failed_calc = False  # should we retry images that failed before? (for calc loop)
retry_invalid_sat = True  # should we retry images that were invalid before (for sat loop)
retry_invalid_sat_est = True  # should we retry images that were invalid before (for sat loop)
retry_invalid_img = False  # # should we retry images that were invalid before (for img loop)
retry_invalid_calc = False  # should we retry images that were invalid before (for calc loop)
retry_no_neighbours_img = True
retry_not_enough_images_calc = True

# some other settings
use_avg_values_approx = True  # can we use avera values for footprint approx
approx_buffer_val = 2000  # how many meters should the approx approx_footprint be buffered in each direction
min_complexity = 0.05

_catch = True
_verbose = True

debug_print = False  # some extra messages
debug_print_all = False  # if true, we print everything on a new line

# for 0029 text must be top

debug_ids = None
debug_starting_id = 2474


# debug_ids = ["CA182332V0023", "CA182332V0025"]
# debug_ids = ["CA184632V0332"]
# debug_ids = ["CA184632V0332"]

def georef_all2(catch=True, verbose=False):
    # get all vertical images and their data from the database
    sql_string_images = f"SELECT * FROM images WHERE image_id LIKE '%V%'"
    data_images = ctd.get_data_from_db(sql_string_images, catch=False, verbose=verbose)

    if debug_ids is None:
        filter_string = f"image_id LIKE '%V%'"
    else:
        formatted_list = "(" + ", ".join(["'{}'".format(item) for item in debug_ids]) + ")"  # noqa
        filter_string = f"image_id in {formatted_list}"

    sql_string_extracted = "SELECT image_id, height, focal_length, complexity, " \
                           "ST_AsText(footprint_approx) AS footprint_approx, " \
                           "ST_AsText(position_approx) AS photocenter_approx, " \
                           "ST_AsText(footprint_exact) AS footprint_exact, " \
                           "ST_AsText(position_exact) AS photocenter, " \
                           "footprint_type AS footprint_type " \
                           f"FROM images_extracted WHERE {filter_string}"
    data_extracted = ctd.get_data_from_db(sql_string_extracted, catch=False, verbose=verbose)

    # merge the data from both tables
    data = pd.merge(data_images, data_extracted, on='image_id')

    # save all image ids in a list
    image_ids = data['image_id'].values.tolist()

    # lists just for displaying reasons
    lists = {
        "georef_sat": [],
        "georef_img": [],
        "georef_calc": [],
        "failed": [],
        "invalid": [],
        "too_few_tps": [],
        "complexity": [],
        "no_neighbours": [],
        "not_enough_images": []
    }

    # the global base path
    base_path = "/data_1/ATM/data_1/playground/georef4"

    # temp folder for storing temporary stuff (I'm Captain obvious, I know)
    temp_path = f"{base_path}/_temp"

    # This dict contains all paths
    path_dict = {
        "approx_shape_footprints": f"{base_path}/overview/approx/approx_footprints.shp",
        "approx_shape_photocenter": f"{base_path}/overview/approx/approx_photocenters.shp",
        "calc_csv_invalid": f"{base_path}/overview/calc_invalid.csv",
        "calc_csv_failed": f"{base_path}/overview/calc_failed.csv",
        "calc_csv_not_enough_images": f"{base_path}/overview/calc_not_enough_images.csv",
        "calc_shape_error_vectors": f"{base_path}/overview/calc/calc_error_vectors.shp",
        "calc_shape_error_vectors_invalid": f"{base_path}/overview/calc_invalid/calc_error_vectors_invalid.shp",
        "calc_shape_footprints": f"{base_path}/overview/calc/calc_footprints.shp",
        "calc_shape_footprints_invalid": f"{base_path}/overview/calc_invalid/calc_footprints_invalid.shp",
        "calc_shape_photocenters": f"{base_path}/overview/calc/calc_photocenters.shp",
        "calc_shape_photocenters_invalid": f"{base_path}/overview/calc_invalid/calc_photocenters_invalid.shp",
        "calc_tiff": f"{base_path}/tiffs/calc/$image_id$.tif",
        "calc_tiff_invalid": f"{base_path}/tiffs/calc_invalid/$image_id$.tif",
        "img_csv_invalid": f"{base_path}/overview/img_invalid.csv",
        "img_csv_too_few_tps": f"{base_path}/overview/img_too_few_tps.csv",
        "img_csv_failed": f"{base_path}/overview/img_failed.csv",
        "img_csv_no_neighbours": f"{base_path}/overview/img_no_neighbours.csv",
        "img_shape_error_vectors": f"{base_path}/overview/img/img_error_vectors.shp",
        "img_shape_error_vectors_invalid": f"{base_path}/overview/img_invalid/img_error_vectors_invalid.shp",
        "img_shape_footprints": f"{base_path}/overview/img/img_footprints.shp",
        "img_shape_footprints_invalid": f"{base_path}/overview/img_invalid/img_footprints_invalid.shp",
        "img_shape_photocenters": f"{base_path}/overview/img/img_photocenters.shp",
        "img_shape_photocenters_invalid": f"{base_path}/overview/img_invalid/img_photocenters_invalid.shp",
        "img_tie_points_shape": f"{base_path}/tie_points/shapes/img/$image_id$_points.shp",
        "img_tie_points_shape_invalid": f"{base_path}/tie_points/shapes/img_invalid/$image_id$_points.shp",
        "img_tie_points_shape_too_few_tps": f"{base_path}/tie_points/shapes/img_too_few_tps/$image_id$_points.shp",
        "img_tiff": f"{base_path}/tiffs/img/$image_id$.tif",
        "img_tiff_invalid": f"{base_path}/tiffs/img_invalid/$image_id$.tif",
        "sat_csv_invalid": f"{base_path}/overview/sat_invalid.csv",
        "sat_csv_too_few_tps": f"{base_path}/overview/sat_too_few_tps.csv",
        "sat_csv_failed": f"{base_path}/overview/sat_failed.csv",
        "sat_shape_error_vectors": f"{base_path}/overview/sat/sat_error_vectors.shp",
        "sat_shape_error_vectors_invalid": f"{base_path}/overview/sat_invalid/sat_error_vectors_invalid.shp",
        "sat_shape_footprints": f"{base_path}/overview/sat/sat_footprints.shp",
        "sat_shape_footprints_invalid": f"{base_path}/overview/sat_invalid/sat_footprints_invalid.shp",
        "sat_shape_photocenters": f"{base_path}/overview/sat/sat_photocenters.shp",
        "sat_shape_photocenters_invalid": f"{base_path}/overview/sat_invalid/sat_photocenters_invalid.shp",
        "sat_tie_points_img": f"{base_path}/tie_points/images/sat/$image_id$_points.png",
        "sat_tie_points_img_invalid": f"{base_path}/tie_points/images/sat_invalid/$image_id$_points.png",
        "sat_tie_points_img_too_few_tps": f"{base_path}/tie_points/images/sat_too_few_tps/$image_id$_points.png",
        "sat_tie_points_shape": f"{base_path}/tie_points/shapes/sat/$image_id$_points.shp",
        "sat_tie_points_shape_invalid": f"{base_path}/tie_points/shapes/sat_invalid/$image_id$_points.shp",
        "sat_tie_points_shape_too_few_tps": f"{base_path}/tie_points/shapes/sat_too_few_tps/$image_id$_points.shp",
        "sat_tiff": f"{base_path}/tiffs/sat/$image_id$.tif",
        "sat_tiff_invalid": f"{base_path}/tiffs/sat_invalid/$image_id$.tif",
        "temp_fld": f"{temp_path}",
        "georef_fld": f"{base_path}/tiffs/sat"
    }

    # some global csv paths
    path_time_csv_sat = f"{base_path}/sat_calc_time.csv"
    path_invalid_footprints_csv = f"{base_path}/sat_invalid_flightpaths.csv"  # noqa

    # the path to the master csv
    path_master_csv = base_path + "/georef_4.csv"

    # calc approx footprint
    if step_calc_approx_footprint:

        # check if any of the columns is nan (we're saving the loop)
        if not data['footprint_approx'].isna().any() and not data['photocenter_approx'].isna().any() \
                and overwrite_approx_footprint is False:
            p.print_v("All images have approximate footprints", verbose=verbose)
        else:
            tqdm.write("Check approx approx_footprint and photocenter for images:")
            sleep(0.1)  # Introduce a small delay
            for image_id in (pbar := tqdm(image_ids)):

                if debug_print_all:
                    pbar = None

                p.print_v(f"Calculate approx values for {image_id}", verbose=verbose, pbar=pbar)

                # create deep copy from path dict to not change values
                _path_dict = copy.deepcopy(path_dict)
                _path_dict = {key: value.replace('$image_id$', image_id) for key, value in _path_dict.items()}

                # get row from the pandas
                row = data[data['image_id'] == image_id].iloc[0]

                if overwrite_approx_footprint is True or \
                        row['footprint_approx'] is None or row['photocenter_approx'] is None:

                    # for the calculation of the approx approx_footprint we need some more information
                    focal_length = row["focal_length"]
                    height = row["height"]
                    x = row["x_coords"]
                    y = row["y_coords"]
                    view_direction = row["view_direction"]
                    angle = row["azimuth"]

                    # we need an angle (and that we cannot substitute)
                    if angle is None:
                        if catch_approx_footprint:
                            p.print_v(f"{image_id} is missing an angle", verbose=verbose,
                                      pbar=pbar, color="red")
                            continue
                        else:
                            raise ValueError(f"{image_id} is missing an angle")

                    # we need a focal length
                    if focal_length is None:
                        if use_avg_values_approx is False:
                            if catch_approx_footprint:
                                p.print_v(f"{image_id} is missing a focal length", verbose=verbose,
                                          pbar=pbar, color="red")
                                continue
                            else:
                                raise ValueError(f"{image_id} is missing a focal length")
                        else:
                            focal_length = 152.457

                    # we need a height
                    if height is None or np.isnan(height):
                        if use_avg_values_approx is False:
                            if catch_approx_footprint:
                                p.print_v(f"{image_id} is missing a height", verbose=verbose,
                                          pbar=pbar, color="red")
                                continue
                            else:
                                raise ValueError(f"{image_id} is missing a height")
                        else:
                            height = 25000

                    # calculate an approximate approx_footprint based on the information we have from the shapefile
                    footprint_approx = cfa.calc_footprint_approx(x, y, angle, view_direction,
                                                                 height, focal_length,
                                                                 catch=True, verbose=False, pbar=pbar)

                    # sometimes we cannot calculate the approx approx_footprint
                    if footprint_approx is None:
                        if catch_approx_footprint:
                            p.print_v(f"For {image_id} no approx approx_footprint could be calculated", verbose=verbose,
                                      pbar=pbar, color="red")
                            continue
                        else:
                            raise ValueError(f"For {image_id} no approx approx_footprint could be calculated")

                    # calculate Photocenter from the approx_footprint
                    x, y = ccp.calc_camera_position(footprint_approx, catch=True)
                    if x is None or y is None:
                        if catch_approx_footprint:
                            p.print_v(f"For {image_id} no approx x and y could be calculated", verbose=verbose,
                                      pbar=pbar, color="red")
                            continue
                        else:
                            raise ValueError(f"For {image_id} no approx approx x and y could be calculated")

                    # create a shapely point that we can save in a db
                    photocenter_approx = Point(x, y)

                    # sometimes we cannot calculate the approx photocenter
                    if photocenter_approx is None:
                        if catch_approx_footprint:
                            p.print_v(f"For {image_id} no approx photocenter could be calculated", verbose=verbose,
                                      pbar=pbar, color="red")
                            continue
                        else:
                            raise ValueError(f"For {image_id} no approx photocenter could be calculated")

                    # Create a GeoDataFrame with the approx_footprint
                    footprint_gdf = gpd.GeoDataFrame(geometry=[footprint_approx])
                    footprint_gdf.crs = "EPSG:3031"
                    footprint_gdf['image_id'] = image_id
                    footprint_gdf['complexity'] = data['complexity']

                    # Create a GeoDataFrame with the photocenter
                    photocenter_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x], [y]))
                    photocenter_gdf.crs = "EPSG:3031"
                    photocenter_gdf['image_id'] = image_id

                    # save approx approx_footprint
                    if not os.path.exists(_path_dict["approx_shape_footprints"]):
                        footprint_gdf.to_file(_path_dict["approx_shape_footprints"])
                    else:
                        # Read the existing shapefile
                        existing_gdf = gpd.read_file(_path_dict["approx_shape_footprints"])

                        # check if the image_id is already in our existing footprints
                        image_id = footprint_gdf.iloc[0]['image_id']
                        if image_id in existing_gdf['image_id'].values:
                            # Update the entry in existing_gdf
                            existing_gdf.loc[existing_gdf['image_id'] == image_id,
                                             "geometry"] = footprint_gdf.iloc[0]['geometry']
                        else:
                            # Append the new approx_footprint to the existing GeoDataFrame
                            existing_gdf = pd.concat([existing_gdf, footprint_gdf], ignore_index=True)

                        # Save the updated GeoDataFrame to the shapefile
                        existing_gdf.to_file(_path_dict["approx_shape_footprints"])

                    # save approx photocenter
                    if not os.path.exists(_path_dict["approx_shape_photocenter"]):
                        photocenter_gdf.to_file(_path_dict["approx_shape_photocenter"])
                    else:
                        # Read the existing shapefile
                        existing_gdf = gpd.read_file(_path_dict["approx_shape_photocenter"])

                        # Check if the image_id is already in our existing photocenters
                        if image_id in existing_gdf['image_id'].values:
                            # Update the entry in existing_gdf
                            existing_gdf.loc[existing_gdf['image_id'] == image_id,
                                             "geometry"] = photocenter_gdf.iloc[0]['geometry']
                        else:
                            # Append the new photocenter to the existing GeoDataFrame
                            existing_gdf = pd.concat([existing_gdf, photocenter_gdf], ignore_index=True)

                        # Save the updated GeoDataFrame to the shapefile
                        existing_gdf.to_file(_path_dict["approx_shape_photocenter"])

                    # save both approx_footprint and photocenter in db
                    sql_string = f"UPDATE images_extracted SET " \
                                 f"footprint_approx=ST_GeomFromText('{footprint_approx}'), " \
                                 f"position_approx=St_GeomFromText('{photocenter_approx}') " \
                                 f"WHERE image_id='{image_id}'"
                    success = ctd.edit_data_in_db(sql_string, catch=True, verbose=False, pbar=None)

                    if success is False:
                        if catch_approx_footprint:
                            p.print_v(
                                f"Something went wrong saving the approx approx_footprint and photocenter in db "
                                f"for {image_id}",
                                verbose=verbose, pbar=pbar, color="red")
                            continue
                        else:
                            raise ValueError(
                                f"Something went wrong saving the approx approx_footprint and photocenter in "
                                f"db for {image_id}")

                    # finally save in the geopandas dataframe that we can already use our
                    # approx_footprint and photocenter
                    data.loc[data['image_id'] == image_id, 'footprint_approx'] = footprint_approx.wkt
                    data.loc[data['image_id'] == image_id, 'photocenter_approx'] = photocenter_approx.wkt

    # remove all images where approx footprint is none
    data = data.dropna(subset=['footprint_approx'])
    image_ids = data['image_id'].values.tolist()

    # georef sat (basic)
    if step_georef_with_satellite_basic:
        tqdm.write("Geo-reference with satellite images:")
        sleep(0.1)  # Introduce a small delay
        counter = 0
        for image_id in (pbar := tqdm(image_ids)):

            counter = counter + 1
            if debug_starting_id is not None and counter < debug_starting_id:
                continue

            if debug_print_all:
                pbar = None

            start = time.time()

            len_georef_sat = len(lists["georef_sat"])
            len_georef_img = len(lists["georef_img"])
            len_georef_calc = len(lists["georef_calc"])
            len_failed = len(lists["failed"])
            len_invalid = len(lists["invalid"])
            len_too_few_tps = len(lists["too_few_tps"])
            len_complexity = len(lists["complexity"])

            p.print_v(f"Work on {image_id}: {len_georef_sat}/{len_georef_img}/{len_georef_calc} "
                      f"georeferenced, {len_failed} failed, "
                      f"{len_invalid} invalid, {len_too_few_tps} too few_tps, "
                      f"{len_complexity} too simple", pbar=pbar)

            # create deep copy from path dict to not change values
            _path_dict = copy.deepcopy(path_dict)
            _path_dict = {key: value.replace('$image_id$', image_id) for key, value in _path_dict.items()}

            # get row from the pandas
            row = data[data['image_id'] == image_id].iloc[0]

            if debug_print:
                p.print_v("Start checking")

            # check if we need to work with image id
            if row['footprint_exact'] is not None and row['footprint_type'] == "satellite":
                if overwrite_satellite_footprint is False:
                    p.print_v(f"{image_id} is already geo-referenced",
                              color="green", verbose=verbose, pbar=pbar)
                    lists['georef_sat'].append(image_id) if image_id not in lists['georef_sat'] else None
                    # end = time.time()  no need to save time here
                    continue

            if debug_print:
                p.print_v("georef passed")

            # Check if image is complex enough to georeference
            if row["complexity"] <= min_complexity:
                p.print_v(f"{image_id} is too simple",
                          color="red", verbose=verbose, pbar=pbar)
                lists['complexity'].append(image_id) if image_id not in lists['complexity'] else None
                end = time.time()
                calc_time = end - start
                _save_to_master_csv(path_master_csv, image_id, "sat", "complexity",
                                    reason="too low",
                                    calc_time=calc_time,
                                    complexity=row["complexity"])
                continue

            if debug_print:
                p.print_v("complexity passed")

            # Check if the image did fail previously
            if retry_failed_sat is False and mc.modify_csv(_path_dict["sat_csv_failed"], image_id, "check"):
                p.print_v(f"{image_id} did previously already fail",
                          color="red", verbose=verbose, pbar=pbar)
                # end = time.time()  no need to save time here
                lists['failed'].append(image_id) if image_id not in lists['failed'] else None
                continue

            if debug_print:
                p.print_v("failed passed")

            # Check if the image had too few tps
            if retry_too_few_tps_sat is False and \
                    mc.modify_csv(_path_dict['sat_csv_too_few_tps'], image_id, "check"):
                p.print_v(f"{image_id} had previously too few tps",
                          color="red", verbose=verbose, pbar=pbar)
                lists['too_few_tps'].append(image_id) if image_id not in lists['too_few_tps'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("too_few_tps passed")

            # Check if the image is in the invalid fld
            if retry_invalid_sat is False and os.path.isfile(_path_dict["sat_tiff_invalid"]):
                lists['invalid'].append(image_id) if image_id not in lists['invalid'] else None
                p.print_v(f"{image_id} was previously invalid",
                          color="red", verbose=verbose, pbar=pbar)
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("invalid passed")

            # get the required data from sql for geo-referencing
            footprint_approx = shapely.wkt.loads(row['footprint_approx'])
            photocenter_approx = shapely.wkt.loads(row['photocenter_approx'])

            approx_x, approx_y = photocenter_approx.x, photocenter_approx.y
            azimuth = row['azimuth']

            # buffer the approx approx_footprint
            footprint_buffered = bf.buffer_footprint(footprint_approx, buffer_val=approx_buffer_val,
                                                     catch=True, verbose=False)

            if footprint_buffered is None:
                # add to failed csv
                mc.modify_csv(_path_dict["sat_csv_failed"], image_id, "add",
                              data={'reason': "buffer_approx_footprint"})

                if catch:
                    p.print_v(f"Failed to buffer the approx approx_footprint for {image_id}",
                              color="red", verbose=verbose, pbar=pbar)
                    lists['failed'].append(image_id) if image_id not in lists['failed'] else None
                    end = time.time()
                    exec_time = end - start
                    mc.modify_csv(path_time_csv_sat, image_id, "add",
                                  data={'time': exec_time})
                    continue
                else:
                    raise ValueError(f"Failed to buffer the approx approx_footprint for {image_id}")

            # now it's time georeference the image
            georef_sat_tuple = gs.georef_sat(image_id, _path_dict["temp_fld"],
                                             footprint_buffered, azimuth,
                                             catch=catch, verbose=False, pbar=pbar)

            footprint_exact = georef_sat_tuple[0]
            transform = georef_sat_tuple[1]
            status = georef_sat_tuple[2]
            residuals = georef_sat_tuple[3]
            nr_tps = georef_sat_tuple[4]

            if debug_print:
                p.print_v("footprint_exact:", footprint_exact)
                p.print_v("transform:", transform)
                p.print_v("status:", status)

            # if we have an approx_footprint (being right or invalid), we can calculate a photocenter and error vector
            if footprint_exact is not None:

                # calculate exact x and y of photocenter
                exact_x, exact_y = ccp.calc_camera_position(footprint_exact)

                # create a shapely point that we can save in a db
                photocenter_exact = Point(exact_x, exact_y)

                # create error vector
                error_vector_exact = LineString([(exact_x, exact_y), (approx_x, approx_y)])

                dx = exact_x - approx_x
                dy = exact_y - approx_y

            else:
                photocenter_exact = None
                error_vector_exact = None
                dx = None
                dy = None

            # remove from all other folders
            data, lists = _clean_image_id(image_id, _path_dict, data, lists, ["sat", "img", "calc"])

            # add to the right folder
            data, lists, status, reason = _put_image_id_right(image_id, "sat", _path_dict, data, lists,
                                                              footprint_exact, photocenter_exact,
                                                              error_vector_exact, dx, dy,
                                                              status, catch, verbose, pbar)

            # remove from temp folder
            data, lists = _clean_image_id(image_id, _path_dict, data, lists, ["temp"])

            end = time.time()
            exec_time = end - start
            mc.modify_csv(path_time_csv_sat, image_id, "add",
                          data={'time': exec_time})

            _save_to_master_csv(path_master_csv, image_id, "sat", status, reason,
                                complexity=row["complexity"], nr_tps=nr_tps,
                                calc_time=exec_time, residuals=residuals, transform=transform)

    # georef sat (estimated)
    if step_georef_with_satellite_estimated:
        tqdm.write("Geo-reference with satellite images and estimated positions:")
        sleep(0.1)  # Introduce a small delay

        # extra list for this iteration
        lst_no_estimated_footprints = []

        counter = 0
        for image_id in (pbar := tqdm(image_ids)):

            counter = counter + 1
            if debug_starting_id is not None and counter < debug_starting_id:
                continue

            if debug_print_all:
                pbar = None

            start = time.time()

            len_georef_sat = len(lists["georef_sat"])
            len_georef_img = len(lists["georef_img"])
            len_georef_calc = len(lists["georef_calc"])
            len_failed = len(lists["failed"])
            len_no_footprint = len(lst_no_estimated_footprints)
            len_invalid = len(lists["invalid"])
            len_too_few_tps = len(lists["too_few_tps"])
            len_complexity = len(lists["complexity"])

            p.print_v(f"Work on {image_id}: {len_georef_sat}/{len_georef_img}/{len_georef_calc} "
                      f"georeferenced, {len_failed} failed, {len_no_footprint} no footprint, "
                      f"{len_invalid} invalid, {len_too_few_tps} too few_tps, "
                      f"{len_complexity} too simple", pbar=pbar)

            # create deep copy from path dict to not change values
            _path_dict = copy.deepcopy(path_dict)
            _path_dict = {key: value.replace('$image_id$', image_id) for key, value in _path_dict.items()}

            # get row from the pandas
            row = data[data['image_id'] == image_id].iloc[0]

            if debug_print:
                p.print_v("Start checking")

            # check if we need to work with image id
            if row['footprint_exact'] is not None and row['footprint_type'] == "satellite":
                if overwrite_satellite_footprint is False:
                    p.print_v(f"{image_id} is already geo-referenced (satellite)",
                              color="green", verbose=verbose, pbar=pbar)
                    lists['georef_sat'].append(image_id) if image_id not in lists['georef_sat'] else None
                    # end = time.time()  no need to save time here
                    continue

            if debug_print:
                p.print_v("georef passed")

            # Check if image is complex enough to georeference
            if row["complexity"] <= min_complexity:
                p.print_v(f"{image_id} is too simple",
                          color="red", verbose=verbose, pbar=pbar)
                lists['complexity'].append(image_id) if image_id not in lists['complexity'] else None
                end = time.time()
                calc_time = end - start
                _save_to_master_csv(path_master_csv, image_id, "sat", "complexity",
                                    calc_time=calc_time,
                                    complexity=row["complexity"])
                continue

            if debug_print:
                p.print_v("complexity passed")

            # Check if the image did fail previously
            if retry_failed_sat_est is False and mc.modify_csv(_path_dict["sat_csv_failed"], image_id, "check"):
                p.print_v(f"{image_id} did previously already fail",
                          color="red", verbose=verbose, pbar=pbar)
                lists['failed'].append(image_id) if image_id not in lists['failed'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("failed passed")

            # Check if the image had too few tps
            if retry_too_few_tps_sat_est is False and \
                    mc.modify_csv(_path_dict['sat_csv_too_few_tps'], image_id, "check"):
                p.print_v(f"{image_id} had previously too few tps",
                          color="red", verbose=verbose, pbar=pbar)
                lists['too_few_tps'].append(image_id) if image_id not in lists['too_few_tps'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("too_few_tps passed")

            # Check if the image is in the invalid fld
            if retry_invalid_sat_est is False and os.path.isfile(_path_dict["sat_tiff_invalid"]):
                p.print_v(f"{image_id} was previously invalid",
                          color="red", verbose=verbose, pbar=pbar)
                lists['invalid'].append(image_id) if image_id not in lists['invalid'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("invalid passed")

            # get the required data from sql for geo-referencing
            photocenter_approx = shapely.wkt.loads(row['photocenter_approx'])
            approx_x, approx_y = photocenter_approx.x, photocenter_approx.y
            azimuth = row['azimuth']

            # estimate a footprint for the image based on already geo-referenced images
            footprint_estimated = dip.derive_image_position(image_id, min_nr_of_images=2,
                                                            catch=catch)

            # check if we got a footprint
            if footprint_estimated[1] is None:
                lst_no_estimated_footprints.append(image_id) if \
                    image_id not in lst_no_estimated_footprints else None

                # add to list of invalid flight paths
                flight_path = image_id[2:6]
                mc.modify_csv(path_invalid_footprints_csv, flight_path, "add")

                p.print_v(f"No footprint could be estimated for {image_id}",
                          color="red", verbose=verbose, pbar=pbar)
                continue

            # now it's time georeference the image with the estimated image
            georef_sat_tuple = gs.georef_sat(image_id, _path_dict["temp_fld"],
                                             footprint_estimated[1], azimuth,
                                             locate_image=False,
                                             catch=catch, verbose=False, pbar=pbar)

            footprint_exact = georef_sat_tuple[0]
            transform = georef_sat_tuple[1]
            status = georef_sat_tuple[2]
            residuals = georef_sat_tuple[3]
            nr_tps = georef_sat_tuple[4]

            # if we have an approx_footprint (being right or invalid), we can calculate a photocenter and error vector
            if footprint_exact is not None:

                # calculate exact x and y of photocenter
                exact_x, exact_y = ccp.calc_camera_position(footprint_exact)

                # create a shapely point that we can save in a db
                photocenter_exact = Point(exact_x, exact_y)

                # create error vector
                error_vector_exact = LineString([(exact_x, exact_y), (approx_x, approx_y)])

                dx = exact_x - approx_x
                dy = exact_y - approx_y

            else:
                photocenter_exact = None
                error_vector_exact = None
                dx = None
                dy = None

            # remove from all other folders
            data, lists = _clean_image_id(image_id, _path_dict, data, lists, ["sat", "img", "calc"])

            # add to the right folder
            data, lists, status, reason = _put_image_id_right(image_id, "sat", _path_dict, data, lists,
                                                              footprint_exact, photocenter_exact,
                                                              error_vector_exact, dx, dy,
                                                              status, catch, verbose, pbar)

            # remove from temp folder
            data, lists = _clean_image_id(image_id, _path_dict, data, lists, ["temp"])

            end = time.time()
            exec_time = end - start
            mc.modify_csv(path_time_csv_sat, image_id, "add",
                          data={'time': exec_time})

            _save_to_master_csv(path_master_csv, image_id, "sat", status, reason,
                                complexity=row["complexity"], nr_tps=nr_tps,
                                calc_time=exec_time, residuals=residuals, transform=transform)

    # georef img
    if step_georef_with_image:
        tqdm.write("Geo-reference with geo-referenced images:")
        sleep(0.1)  # Introduce a small delay

        counter = 0
        for image_id in (pbar := tqdm(image_ids)):

            counter = counter + 1
            if debug_starting_id is not None and counter < debug_starting_id:
                continue

            if debug_print_all:
                pbar = None

            start = time.time()

            len_georef_sat = len(lists["georef_sat"])
            len_georef_img = len(lists["georef_img"])
            len_georef_calc = len(lists["georef_calc"])
            len_failed = len(lists["failed"])
            len_invalid = len(lists["invalid"])
            len_too_few_tps = len(lists["too_few_tps"])
            len_no_neighbours = len(lists["no_neighbours"])

            p.print_v(f"Work on {image_id}: {len_georef_sat}/{len_georef_img}/{len_georef_calc} "
                      f"georeferenced, {len_failed} failed, "
                      f"{len_invalid} invalid, {len_too_few_tps} too few_tps, "
                      f"{len_no_neighbours} no neighbours", pbar=pbar)

            # create deep copy from path dict to not change values
            _path_dict = copy.deepcopy(path_dict)
            _path_dict = {key: value.replace('$image_id$', image_id) for key, value in _path_dict.items()}

            # get row from the pandas
            row = data[data['image_id'] == image_id].iloc[0]

            if debug_print:
                p.print_v("Start checking")

            # check if we need to work with image id
            if row['footprint_exact'] is not None and row['footprint_type'] == "satellite":
                p.print_v(f"{image_id} is already geo-referenced (satellite)",
                          color="green", verbose=verbose, pbar=pbar)
                lists['georef_sat'].append(image_id) if image_id not in lists['georef_sat'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("georef (sat) passed")

            if row['footprint_exact'] is not None and row['footprint_type'] == "image":
                if overwrite_image_footprint is False:
                    p.print_v(f"{image_id} is already geo-referenced (img)",
                              color="green", verbose=verbose, pbar=pbar)
                    lists['georef_img'].append(image_id) if image_id not in lists['georef_img'] else None
                    # end = time.time()  no need to save time here
                    continue

            if debug_print:
                p.print_v("georef (img) passed")

            # Check if the image did fail previously
            if retry_failed_img is False and mc.modify_csv(_path_dict["img_csv_failed"], image_id, "check"):
                p.print_v(f"{image_id} did previously already fail",
                          color="red", verbose=verbose, pbar=pbar)
                lists['failed'].append(image_id) if image_id not in lists['failed'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("failed passed")

            # Check if the image had too few tps
            if retry_too_few_tps_img is False and \
                    mc.modify_csv(_path_dict['img_csv_too_few_tps'], image_id, "check"):
                p.print_v(f"{image_id} had previously too few tps",
                          color="red", verbose=verbose, pbar=pbar)
                lists['too_few_tps'].append(image_id) if image_id not in lists['too_few_tps'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("too_few_tps passed")

            # Check if the image is in the invalid fld
            if retry_invalid_img is False and os.path.isfile(_path_dict["img_tiff_invalid"]):
                p.print_v(f"{image_id} was previously invalid",
                          color="red", verbose=verbose, pbar=pbar)
                lists['invalid'].append(image_id) if image_id not in lists['invalid'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("invalid passed")

            # check if the image had previously no neighbours
            if retry_no_neighbours_img is False and \
                    mc.modify_csv(_path_dict['img_csv_no_neighbours'], image_id, "check"):
                p.print_v(f"{image_id} had no geo-referenced neighbours",
                          color="red", verbose=verbose, pbar=pbar)
                lists['no_neighbours'].append(image_id) if image_id not in lists['no_neighbours'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("no neighbours passed")

            # get the required data from sql for geo-referencing
            photocenter_approx = shapely.wkt.loads(row['photocenter_approx'])
            approx_x, approx_y = photocenter_approx.x, photocenter_approx.y

            georef_img_tuple = gi.georef_img(image_id, _path_dict["temp_fld"],
                                             _path_dict["georef_fld"],
                                             catch=catch, verbose=False, pbar=pbar)

            footprint_image = georef_img_tuple[0]
            transform = georef_img_tuple[1]
            status = georef_img_tuple[2]
            nr_tps = georef_img_tuple[3]
            residuals = georef_img_tuple[4]

            # if we have an approx_footprint (being right or invalid), we can calculate a photocenter and error vector
            if footprint_image is not None:

                # calculate exact x and y of photocenter
                image_x, image_y = ccp.calc_camera_position(footprint_image)

                # create a shapely point that we can save in a db
                photocenter_image = Point(image_x, image_y)

                # create error vector
                error_vector_image = LineString([(image_x, image_y), (approx_x, approx_y)])

                dx = image_x - approx_x
                dy = image_y - approx_y

            else:
                photocenter_image = None
                error_vector_image = None
                dx = None
                dy = None

            # remove from all other folders
            data, lists = _clean_image_id(image_id, _path_dict, data, lists, ["img", "calc"])

            # add to the right folder
            data, lists, status, reason = _put_image_id_right(image_id, "img", _path_dict, data, lists,
                                                              footprint_image, photocenter_image,
                                                              error_vector_image, dx, dy,
                                                              status, catch, verbose, pbar)

            # remove from temp folder
            data, lists = _clean_image_id(image_id, _path_dict, data, lists, ["temp"])

            end = time.time()
            exec_time = end - start
            mc.modify_csv(path_time_csv_sat, image_id, "add",
                          data={'time': exec_time})

            _save_to_master_csv(path_master_csv, image_id, "img", status, reason,
                                complexity=row["complexity"], nr_tps=nr_tps,
                                calc_time=exec_time, residuals=residuals, transform=transform)

    # georef_calc
    if step_georef_with_calc:
        tqdm.write("Geo-reference with calculated positions:")
        sleep(0.1)  # Introduce a small delay

        counter = 0
        for image_id in (pbar := tqdm(image_ids)):

            counter = counter + 1
            if debug_starting_id is not None and counter < debug_starting_id:
                continue

            if debug_print_all:
                pbar = None

            start = time.time()

            len_georef_sat = len(lists["georef_sat"])
            len_georef_img = len(lists["georef_img"])
            len_georef_calc = len(lists["georef_calc"])
            len_failed = len(lists["failed"])
            len_invalid = len(lists["invalid"])
            len_too_few_tps = len(lists["too_few_tps"])
            len_complexity = len(lists["complexity"])
            len_not_enough_images = len(lists["not_enough_images"])

            p.print_v(f"Work on {image_id}: {len_georef_sat}/{len_georef_img}/{len_georef_calc} "
                      f"georeferenced, {len_failed} failed, "
                      f"{len_invalid} invalid, {len_too_few_tps} too few_tps, "
                      f"{len_complexity} too simple, "
                      f"{len_not_enough_images} not enough images",
                      pbar=pbar)

            # create deep copy from path dict to not change values
            _path_dict = copy.deepcopy(path_dict)
            _path_dict = {key: value.replace('$image_id$', image_id) for key, value in _path_dict.items()}

            # get row from the pandas
            row = data[data['image_id'] == image_id].iloc[0]

            if debug_print:
                p.print_v("Start checking")

            # check if we need to work with image id
            if row['footprint_exact'] is not None and row['footprint_type'] == "satellite":
                p.print_v(f"{image_id} is already geo-referenced (satellite)",
                          color="green", verbose=verbose, pbar=pbar)
                lists['georef_sat'].append(image_id) if image_id not in lists['georef_sat'] else None
                # end = time.time()  no need to save time here
                continue

            if row['footprint_exact'] is not None and row['footprint_type'] == "image":
                p.print_v(f"{image_id} is already geo-referenced (img)",
                          color="green", verbose=verbose, pbar=pbar)
                lists['georef_img'].append(image_id) if image_id not in lists['georef_img'] else None
                # end = time.time()  no need to save time here
                continue

            if row['footprint_exact'] is not None and row['footprint_type'] == "calc":
                if overwrite_calc_footprint is False:
                    p.print_v(f"{image_id} is already geo-referenced (calc)",
                              color="green", verbose=verbose, pbar=pbar)
                    lists['georef_calc'].append(image_id) if image_id not in lists['georef_calc'] else None
                    # end = time.time()  no need to save time here
                    continue

            if debug_print:
                p.print_v("georef passed")

            # Check if the image did fail previously
            if retry_failed_calc is False and mc.modify_csv(_path_dict["calc_csv_failed"], image_id, "check"):
                p.print_v(f"{image_id} was previously invalid",
                          color="red", verbose=verbose, pbar=pbar)
                lists['failed'].append(image_id) if image_id not in lists['failed'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("failed passed")

            # Check if the image is in the invalid fld
            if retry_invalid_calc is False and os.path.isfile(_path_dict["calc_tiff_invalid"]):
                p.print_v(f"{image_id} was previously invalid",
                          color="red", verbose=verbose, pbar=pbar)
                lists['invalid'].append(image_id) if image_id not in lists['invalid'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("invalid passed")

            # check if the image had previously no neighbours
            if retry_not_enough_images_calc is False and \
                    mc.modify_csv(_path_dict['calc_csv_not_enough_images'], image_id, "check"):
                p.print_v(f"{image_id} had not enough images",
                          color="red", verbose=verbose, pbar=pbar)
                lists['not_enough_images'].append(image_id) if image_id not in lists['not_enough_images'] else None
                # end = time.time()  no need to save time here
                continue

            if debug_print:
                p.print_v("not enough images passed")

            # get the required data from sql for geo-referencing
            photocenter_approx = shapely.wkt.loads(row['photocenter_approx'])
            approx_x, approx_y = photocenter_approx.x, photocenter_approx.y

            # georeference the image by calculation
            georef_calc_tuple = gc.georef_calc(image_id, _path_dict["temp_fld"],
                                               catch=catch, verbose=False, pbar=pbar)

            footprint_calc = georef_calc_tuple[0]
            transform = georef_calc_tuple[1]
            status = georef_calc_tuple[2]
            nr_tps = georef_calc_tuple[3]
            residuals = georef_calc_tuple[4]

            # if we have an approx_footprint (being right or invalid), we can calculate a photocenter and error vector
            if footprint_calc is not None:

                # calculate exact x and y of photocenter
                calc_x, calc_y = ccp.calc_camera_position(footprint_calc)

                # create a shapely point that we can save in a db
                photocenter_calc = Point(calc_x, calc_y)

                # create error vector
                error_vector_calc = LineString([(calc_x, calc_y), (approx_x, approx_y)])

                dx = calc_x - approx_x
                dy = calc_y - approx_y

            else:
                photocenter_calc = None
                error_vector_calc = None
                dx = None
                dy = None

            # remove from all other folders
            data, lists = _clean_image_id(image_id, _path_dict, data, lists, ["calc"])

            # add to the right folder
            data, lists, status, reason = _put_image_id_right(image_id, "calc", _path_dict, data, lists,
                                                              footprint_calc, photocenter_calc,
                                                              error_vector_calc, dx, dy,
                                                              status, catch, verbose, pbar)

            # remove from temp folder
            data, lists = _clean_image_id(image_id, _path_dict, data, lists, ["temp"])

            end = time.time()
            exec_time = end - start
            mc.modify_csv(path_time_csv_sat, image_id, "add",
                          data={'time': exec_time})

            _save_to_master_csv(path_master_csv, image_id, "calc", status, reason,
                                complexity=row["complexity"], nr_tps=nr_tps,
                                calc_time=exec_time, residuals=residuals, transform=transform)


def _put_image_id_right(image_id, georef_type, _path_dict, data, lists,
                        footprint, photocenter, error_vector, dx, dy,
                        status, catch, verbose, pbar):
    # get the path to the temp files
    path_tmp_tiff = _path_dict["temp_fld"] + "/" + image_id + ".tif"
    path_tmp_tps_img = _path_dict["temp_fld"] + "/" + image_id + "_points.png"
    path_tmp_tps_shape = _path_dict["temp_fld"] + "/" + image_id + "_points.shp"

    # for the master csv
    reason = ""

    # check if we have a georeferenced footprint
    if footprint is not None and status.startswith("georeferenced"):

        if os.path.isfile(path_tmp_tiff) is False:
            raise ValueError(f"There is no image at '{path_tmp_tiff}'. Is 'debug_save' True?")

        # let's check if the image is valid
        image_is_valid, invalid_reason = cgi.check_georef_image(path_tmp_tiff, return_reason=True,
                                                                catch=catch, verbose=verbose, pbar=pbar)

        # unfortunately image is not valid -> change status to invalid
        if image_is_valid is False:
            status = f"invalid:{invalid_reason}"
            reason = invalid_reason

    # depending on the outcome of georef_sat, there are different ways to continue
    if status == "georeferenced":

        lists[f"georef_{georef_type}"].append(image_id) if image_id not in lists[f"georef_{georef_type}"] else None

        p.print_v(f"{image_id} is successfully geo-referenced!", color="green",
                  verbose=verbose, pbar=pbar)

        if georef_type == "sat":
            footprint_type = "satellite"
        elif georef_type == "img":
            footprint_type = "image"
        else:
            footprint_type = georef_type

        # update database
        sql_string = f"UPDATE images_extracted SET " \
                     f"footprint_exact=ST_GeomFromText('{footprint.wkt}'), " \
                     f"position_exact=ST_GeomFromText('{photocenter.wkt}'), " \
                     f"position_error_vector='{dx};{dy}', " \
                     f"footprint_type='{footprint_type}' " \
                     f"WHERE image_id='{image_id}'"
        ctd.edit_data_in_db(sql_string, catch=False, verbose=False)

        # update also pandas data
        data.loc[data['image_id'] == image_id, 'footprint_exact'] = footprint.wkt
        data.loc[data['image_id'] == image_id, 'footprint_type'] = footprint_type

        # copy tiff from temp folder to the right folder
        os.rename(path_tmp_tiff, _path_dict[f"{georef_type}_tiff"]) if os.path.exists(path_tmp_tiff) else None

        # update the footprints, photocenters and error vectors
        ms.modify_shape(_path_dict[f"{georef_type}_shape_footprints"], image_id, "add", footprint)
        ms.modify_shape(_path_dict[f"{georef_type}_shape_photocenters"], image_id, "add", photocenter)
        ms.modify_shape(_path_dict[f"{georef_type}_shape_error_vectors"], image_id, "add", error_vector)

        if georef_type != "calc":
            # copy the tie points images and shapes to the right folder
            os.rename(path_tmp_tps_img, _path_dict[f"{georef_type}_tie_points_img"]) if \
                os.path.exists(path_tmp_tps_img) else None

            # Copy each file to the destination
            for filepath in glob.glob(path_tmp_tps_shape[:-4] + ".*"):
                shutil.copy(filepath, _path_dict[f"{georef_type}_tie_points_shape"][:-5] + filepath[-4:])

        reason = ""

    elif status.startswith("too_few_tps"):
        lists["too_few_tps"].append(image_id) if image_id not in lists["too_few_tps"] else None

        p.print_v(f"{image_id} has too few ({status.split(':')[1]}) tps!", color="red",
                  verbose=verbose, pbar=pbar)

        if georef_type != "calc":

            # save in too few tps csv
            mc.modify_csv(_path_dict[f"{georef_type}_csv_too_few_tps"], image_id, "add",
                          data={'tie_points': int(status.split(":")[1])})

            # copy the tie points images and shapes to the right folder
            os.rename(path_tmp_tps_img, _path_dict[f"{georef_type}_tie_points_img_too_few_tps"]) if \
                os.path.exists(path_tmp_tps_img) else None

            # Copy each file to the destination
            for filepath in glob.glob(path_tmp_tps_shape[:-4] + ".*"):
                shutil.copy(filepath, _path_dict[f"{georef_type}_tie_points_shape_too_few_tps"][:-5] + filepath[-4:])

        reason = "too_few_tps"

    elif status.startswith("invalid"):
        lists["invalid"].append(image_id) if image_id not in lists["invalid"] else None

        p.print_v(f"{image_id} is invalid!", color="red",
                  verbose=verbose, pbar=pbar)

        # save in invalid csv
        mc.modify_csv(_path_dict[f"{georef_type}_csv_invalid"], image_id, "add",
                      data={'reason': status.split(":")[1]})

        # copy tiff from temp folder to the right folder
        os.rename(path_tmp_tiff, _path_dict[f"{georef_type}_tiff_invalid"]) if \
            os.path.exists(path_tmp_tiff) else None

        # update the footprints, photocenters and error vectors
        ms.modify_shape(_path_dict[f"{georef_type}_shape_footprints_invalid"],
                        image_id, "add", footprint)
        ms.modify_shape(_path_dict[f"{georef_type}_shape_photocenters_invalid"],
                        image_id, "add", photocenter)
        ms.modify_shape(_path_dict[f"{georef_type}_shape_error_vectors_invalid"],
                        image_id, "add", error_vector)

        # copy the tie points images and shapes to the right folder
        os.rename(path_tmp_tps_img, _path_dict[f"{georef_type}_tie_points_img_invalid"]) if \
            os.path.exists(path_tmp_tps_img) else None

        # Copy each file to the destination
        for filepath in glob.glob(path_tmp_tps_shape[:-4] + ".*"):
            shutil.copy(filepath, _path_dict[f"{georef_type}_tie_points_shape_invalid"][:-5] + filepath[-4:])

        reason = status.split(":")[1]

    elif status.startswith("failed"):
        lists["failed"].append(image_id) if image_id not in lists["failed"] else None

        p.print_v(f"{image_id} failed!", color="red",
                  verbose=verbose, pbar=pbar)

        # save in failed csv
        mc.modify_csv(_path_dict[f"{georef_type}_csv_failed"], image_id, "add",
                      data={'reason': status.split(":")[1]})

        # delete possibly already existing stuff
        os.remove(path_tmp_tiff) if os.path.exists(path_tmp_tiff) else None
        os.remove(path_tmp_tps_img) if os.path.exists(path_tmp_tiff) else None
        for filepath in glob.glob(path_tmp_tps_shape[:-4] + ".*"):
            os.remove(filepath) if os.path.exists(filepath) else None

        reason = status.split(":")[1]

    elif status == "no_neighbours":
        lists["no_neighbours"].append(image_id) if image_id not in lists["no_neighbours"] else None
        p.print_v(f"{image_id} has no neighbours!", color="red",
                  verbose=verbose, pbar=pbar)

        # save in no neighbours csv
        mc.modify_csv(_path_dict[f"img_csv_no_neighbours"], image_id, "add")

        reason = ""

    elif status == "not_enough_images":
        lists["not_enough_images"].append(image_id) if image_id not in lists["not_enough_images"] else None
        p.print_v(f"{image_id} has non enough geo-referenced images on the flight-path!", color="red",
                  verbose=verbose, pbar=pbar)

        # save in not enough images csv
        mc.modify_csv(_path_dict[f"calc_csv_not_enough_images"], image_id, "add")

        reason = ""

    return data, lists, status, reason


def _save_to_master_csv(path_master_csv, image_id, method, status, reason=None,
                        complexity=None, nr_tps=None, calc_time=0.0, residuals=None,
                        transform=None):
    if ":" in status:
        status = status.split(":")[0]

    if reason is None:
        reason = ""

    if complexity is None:
        complexity = ""

    if nr_tps is None:
        nr_tps = ""

    if calc_time is None or calc_time == 0.0:
        calc_time = ""

    if residuals is None:
        residuals = ""

    if transform is None:
        transform = [""] * 9

    # create data dict for master csv
    master_dict = {
        'status': status,
        'reason': reason,
        'method': method,
        'complexity': complexity,
        'nr_tps': nr_tps,
        'calc_time': calc_time,
        'residuals': residuals,
        't0': transform[0],
        't1': transform[1],
        't2': transform[2],
        't3': transform[3],
        't4': transform[4],
        't5': transform[5],
        't6': transform[6],
        't7': transform[7],
        't8': transform[8],
        'timestamp': datetime.now()
    }

    # update the master csv
    mc.modify_csv(path_master_csv, image_id, "add", data=master_dict, overwrite=True)


def _clean_image_id(image_id, _path_dict, data, lists, to_delete):
    # Loop through each file in tmp_folder and remove
    if "temp" in to_delete:
        all_files = os.listdir(_path_dict["temp_fld"])
        for file_name in all_files:
            # Check if the file name contains the substring
            if image_id in file_name:
                # Construct full file path
                file_path = os.path.join(_path_dict["temp_fld"], file_name)
                # Delete the file
                os.remove(file_path)
        return data, lists

    try:
        lists["georef_sat"].remove(image_id)
        lists["georef_img"].remove(image_id)
        lists["georef_calc"].remove(image_id)
        lists["failed"].remove(image_id)
        lists["invalid"].remove(image_id)
        lists["too_few_tps"].remove(image_id)
        lists["complexity"].remove(image_id)
        lists["no_neighbours"].remove(image_id)
        lists["not_enough_images"].remove(image_id)
    except ValueError:
        pass

    # delete the tiffs
    if "sat" in to_delete:
        os.remove(_path_dict["sat_tiff"]) if os.path.exists(_path_dict["sat_tiff"]) else None
        os.remove(_path_dict["sat_tiff_invalid"]) if os.path.exists(_path_dict["sat_tiff_invalid"]) else None
    if "img" in to_delete:
        os.remove(_path_dict["calc_tiff"]) if os.path.exists(_path_dict["calc_tiff"]) else None
        os.remove(_path_dict["calc_tiff_invalid"]) if os.path.exists(_path_dict["calc_tiff_invalid"]) else None
    if "calc" in to_delete:
        os.remove(_path_dict["img_tiff"]) if os.path.exists(_path_dict["img_tiff"]) else None
        os.remove(_path_dict["img_tiff_invalid"]) if os.path.exists(_path_dict["img_tiff_invalid"]) else None

    # remove from the csv files
    mc.modify_csv(_path_dict["sat_csv_invalid"], image_id, "delete")
    mc.modify_csv(_path_dict["sat_csv_too_few_tps"], image_id, "delete")
    mc.modify_csv(_path_dict["sat_csv_failed"], image_id, "delete")
    mc.modify_csv(_path_dict["img_csv_invalid"], image_id, "delete")
    mc.modify_csv(_path_dict["img_csv_too_few_tps"], image_id, "delete")
    mc.modify_csv(_path_dict["img_csv_failed"], image_id, "delete")
    mc.modify_csv(_path_dict["img_csv_no_neighbours"], image_id, "delete")
    mc.modify_csv(_path_dict["calc_csv_failed"], image_id, "delete")
    mc.modify_csv(_path_dict["calc_csv_not_enough_images"], image_id, "delete")

    # remove from the overview shape files
    if "sat" in to_delete:
        ms.modify_shape(_path_dict["sat_shape_error_vectors"], image_id=image_id, modus="delete")
        ms.modify_shape(_path_dict["sat_shape_footprints"], image_id=image_id, modus="delete")
        ms.modify_shape(_path_dict["sat_shape_photocenters"], image_id=image_id, modus="delete")
    if "img" in to_delete:
        ms.modify_shape(_path_dict["img_shape_error_vectors"], image_id=image_id, modus="delete")
        ms.modify_shape(_path_dict["img_shape_footprints"], image_id=image_id, modus="delete")
        ms.modify_shape(_path_dict["img_shape_photocenters"], image_id=image_id, modus="delete")
    if "calc" in to_delete:
        ms.modify_shape(_path_dict["calc_shape_error_vectors"], image_id=image_id, modus="delete")
        ms.modify_shape(_path_dict["calc_shape_footprints"], image_id=image_id, modus="delete")
        ms.modify_shape(_path_dict["calc_shape_photocenters"], image_id=image_id, modus="delete")

    ms.modify_shape(_path_dict["sat_shape_error_vectors_invalid"], image_id=image_id, modus="delete")
    ms.modify_shape(_path_dict["sat_shape_footprints_invalid"], image_id=image_id, modus="delete")
    ms.modify_shape(_path_dict["sat_shape_photocenters_invalid"], image_id=image_id, modus="delete")
    ms.modify_shape(_path_dict["img_shape_error_vectors_invalid"], image_id=image_id, modus="delete")
    ms.modify_shape(_path_dict["img_shape_footprints_invalid"], image_id=image_id, modus="delete")
    ms.modify_shape(_path_dict["img_shape_photocenters_invalid"], image_id=image_id, modus="delete")
    ms.modify_shape(_path_dict["calc_shape_error_vectors_invalid"], image_id=image_id, modus="delete")
    ms.modify_shape(_path_dict["calc_shape_footprints_invalid"], image_id=image_id, modus="delete")
    ms.modify_shape(_path_dict["calc_shape_photocenters_invalid"], image_id=image_id, modus="delete")

    # remove the tie-point shapes
    if "sat" in to_delete:
        ms.modify_shape(_path_dict[f"sat_tie_points_shape"], image_id=None, modus="hard_delete")
    if "img" in to_delete:
        ms.modify_shape(_path_dict[f"img_tie_points_shape"], image_id=None, modus="hard_delete")
    ms.modify_shape(_path_dict[f"sat_tie_points_shape_invalid"], image_id=None, modus="hard_delete")
    ms.modify_shape(_path_dict[f"sat_tie_points_shape_too_few_tps"], image_id=None, modus="hard_delete")
    ms.modify_shape(_path_dict[f"img_tie_points_shape_invalid"], image_id=None, modus="hard_delete")
    ms.modify_shape(_path_dict[f"img_tie_points_shape_too_few_tps"], image_id=None, modus="hard_delete")

    # remove the tie-point images
    if "sat" in to_delete:
        os.remove(_path_dict["sat_tie_points_img"]) if os.path.exists(_path_dict["sat_tie_points_img"]) else None
    os.remove(_path_dict["sat_tie_points_img_invalid"]) if os.path.exists(
        _path_dict["sat_tie_points_img_invalid"]) else None
    os.remove(_path_dict["sat_tie_points_img_too_few_tps"]) if os.path.exists(
        _path_dict["sat_tie_points_img_too_few_tps"]) else None

    sql_string = "UPDATE images_extracted SET footprint_exact = NULL, " \
                 "position_exact = NULL, position_error_vector = NULL, footprint_type = NULL " \
                 f"WHERE image_id='{image_id}'"
    ctd.edit_data_in_db(sql_string, catch=False, verbose=False)

    # data.loc[data['image_id'] == image_id, 'footprint_approx'] = ""
    # data.loc[data['image_id'] == image_id, 'photocenter_approx'] = ""

    return data, lists


if __name__ == "__main__":
    georef_all2(catch=_catch, verbose=_verbose)
