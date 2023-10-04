import copy
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import shapely.wkt
import warnings

from shapely.geometry import Point, LineString

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.modify_shape as ms
import base.modify_csv as mc
import base.print_v as p
import base.remove_borders as rb
import base.rotate_image as ri

import image_georeferencing.sub.adjust_image_resolution as air
import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.buffer_footprint as bf
import image_georeferencing.sub.calc_footprint_approx as cfa
import image_georeferencing.sub.calc_camera_position as ccp
import image_georeferencing.sub.check_georef_image as cgi
import image_georeferencing.sub.convert_image_to_footprint as citf
import image_georeferencing.sub.enhance_image as eh
import image_georeferencing.sub.filter_tp_outliers as fto
import image_georeferencing.sub.find_footprint_direction as ffd
import image_georeferencing.sub.load_satellite_data as lsd

import image_tie_points.find_tie_points as ftp

import display.display_images as di
import display.display_tiepoints as dt

# general settings
_overwrite = True
_catch = False
_verbose = True

# function settings
buffer_val = 2000  # how many meters should the approx approx_footprint be buffered in each direction
enhancement_min = 0.02
enhancement_max = 0.98
error_vector_calculation = "both"  # can be 'closest' or 'avg' or 'both'
error_vector_order = 3  # how many of the closest error vectors should be taken?
filter_outliers = True  # should we filter for outliers in the tps with ransac
location_max_order = 3  # how many layers we want to use (at debug_locate)
location_overlap = 1 / 3  # how much overlap do we want to have between images (at debug_locate)
location_success_criteria = "points"  # how do we define success (at debug_locate; "points" or "conf")
min_complexity = 0.05  # the minimum complexity of an image that it can be used for geo-referencing
min_threshold = 0.8  # the minimum confidence of tie-points
min_nr_of_tps_running = 5  # when do we skip further looking
min_nr_of_tps_final = 25  # how many tie-points we need minimum for georef
max_nr_stop_combination = 100 # if we've found so many points at least, we can stop checking for combinations
transform_method = "rasterio"  # should 'rasterio' or 'gdal' be used for geo-referencing
transform_order = 1  # which order of transform should be used
tweaking_max_iterations = 10  # how often can we tweak

# function params
debug_adapt_sat_bounds_with_existing = True  # should we adapt the sat bounds based on images from same flightpath
debug_check_georef = True  # should geo-referenced images be checked for validity
debug_enhance_images = True  # should images be enhanced before geo-referencing them
debug_georeference = True  # should images be geo-referenced
debug_recalculate_tie_points = True
debug_tweak_image = True  # should we tweak images to get more tps?
debug_use_avg_values = True  # should we use average values for focal length and height if not existing
debug_use_month = False  # If true we use the exact month of an image for the sat image
debug_use_small_image_size = False  # if true images are geo-referenced with an adapted pixel size

# show params
debug_show_initial_images = False  # shows the image and the initial satellite image
debug_show_tie_points_final = True  # shows the final tie-points used for geo-referencing
debug_show_tie_points_final_unfiltered = True  # shows the final tie-points before filtering
debug_show_tie_points_initial = True  # shows the first tie-points from the approximate location
debug_show_tie_points_located = True  # shows the tie-points after location finding
debug_show_tie_points_tweaked = True  # shows the tie-points after tweaking of the images

# print params
debug_print_used_satellite_img = False

# save params
debug_save_approx_footprints_shape = True  # save approx footprints in an overview shape
debug_save_approx_photocenters_shape = True  # save approx photocenters in an overview shape
debug_save_error_vectors_shape = True  # save the error-vector between approx and exact in an overview shape
debug_save_exact_footprints_shape = True  # save exact footprints in an overview shape
debug_save_exact_photocenters_shape = True  # save exact photocenters in an overview shape
debug_save_tie_points_image = True  # save images where we can see the tie-points
debug_save_tie_points_shape = True  # save the tie-points as a shapefile

# here we want to store the created data
base_path = "/data_1/ATM/data_1/playground/georef3"

debug_located_coords = None


def georef_sat(image_id, existing_footprint=None,
               retry_too_few_tps=False, retry_failed=False, retry_invalid=False,
               recalculate_approx_footprint=False,
               locate_image=True,
               save_in_db=True,
               overwrite=False, catch=True, verbose=False, pbar=None):
    """

    Args:
        image_id:
        existing_footprint:
        retry_too_few_tps:
        retry_failed:
        retry_invalid:
        locate_image: If true we look around the approx footprints for other tie-points
        overwrite:
        catch:
        verbose:
        pbar:

    Returns:

    """

    # overview paths
    path_approx_footprints_shape = base_path + "/overview/approx/approx_footprints.shp"
    path_approx_photocenters_shape = base_path + "/overview/approx/approx_photocenters.shp"
    path_exact_footprints_shape = base_path + "/overview/sat/sat_footprints.shp"
    path_exact_footprints_shape_invalid = base_path + "/overview/sat_invalid/sat_footprints_invalid.shp"
    path_exact_photocenters_shape = base_path + "/overview/sat/sat_photocenters.shp"
    path_exact_photocenters_shape_invalid = base_path + "/overview/sat_invalid/sat_photocenters_invalid.shp"
    path_error_vector_shape = base_path + "/overview/sat/sat_error_vectors.shp"
    path_error_vector_shape_invalid = base_path + "/overview/sat_invalid/sat_error_vectors_invalid.shp"

    # path for images with too few tps or failed
    path_too_few_tps_csv = base_path + "/overview/sat_tie_points.csv"
    path_failed_csv = base_path + "/overview/sat_failed.csv"

    p.print_v(f"georeference {image_id} based on satellite", verbose=verbose, pbar=pbar)

    # tiff paths
    path_tiff = base_path + f"/tiffs/sat/{image_id}.tif"
    path_tiff_invalid = base_path + f"/tiffs/sat_invalid/{image_id}.tif"

    # tie point paths
    path_tie_points_img = base_path + f"/tie_points/images/sat/{image_id}.png"
    path_tie_points_img_invalid = base_path + f"/tie_points/images/sat_invalid/{image_id}.png"
    path_tie_points_img_too_few_tps = base_path + f"/tie_points/images/sat_too_few_tps/{image_id}.png"
    path_tie_points_shp = base_path + f"/tie_points/shapes/sat/{image_id}_points.shp"
    path_tie_points_shp_invalid = base_path + f"/tie_points/shapes/sat_invalid/{image_id}_points.shp"
    path_tie_points_shp_too_few_tps = base_path + f"/tie_points/shapes/sat_too_few_tps/{image_id}.png"

    # get information from table images
    sql_string = f"SELECT * FROM images WHERE image_id='{image_id}'"
    data_images = ctd.get_data_from_db(sql_string, catch=False, verbose=verbose)

    # get information from table images_extracted
    sql_string = "SELECT image_id, height, focal_length, complexity, " \
                 "ST_AsText(footprint_approx) AS footprint_approx, " \
                 "ST_AsText(footprint) AS footprint, " \
                 "footprint_type AS footprint_type " \
                 f"FROM images_extracted WHERE image_id ='{image_id}'"
    data_extracted = ctd.get_data_from_db(sql_string, catch=False, verbose=verbose)

    # merge the data from both tables
    data = pd.merge(data_images, data_extracted, on='image_id').iloc[0]

    # get information if we already did something with the image
    previous_georef_type = data['footprint_type']

    # First, we check if the image is already georeferenced
    if overwrite is False and data['footprint'] is not None and \
            previous_georef_type == "satellite":
        p.print_v(f"{image_id} is already georeferenced!", verbose=verbose,
                  pbar=pbar, color="green")
        return "georeferenced"

    # Second, we check if image is complex enough to georeference
    if data["complexity"] <= min_complexity:
        p.print_v(f"{image_id} is too simple for geo-referencing ({data['complexity']})",
                  verbose=verbose, pbar=pbar, color="red")
        return "complexity"

    # Third we check if the image had too few tie-points
    if retry_too_few_tps is False and mc.modify_csv(path_too_few_tps_csv, image_id, "check"):
        p.print_v(f"{image_id} had too few tie-points", verbose=verbose, pbar=pbar, color="red")
        return "too_few_tps"

    # Fourth, we check if the image is in the invalid fld
    if os.path.isfile(path_tiff_invalid):

        # we don't want to try them again -> continue
        if retry_invalid is False:
            p.print_v(f"{image_id} was already invalid for geo-referencing",
                      verbose=verbose, pbar=pbar, color="red")
            return "invalid"

        # we want to try them again -> remove from invalid
        else:

            # remove the tiff
            os.remove(path_tiff_invalid) if os.path.exists(path_tiff_invalid) else None

            # remove the invalid tie-point file
            ms.modify_shape(path_tie_points_shp_invalid, image_id=None, modus="hard_delete")
            os.remove(path_tie_points_img_invalid) if os.path.exists(path_tie_points_img_invalid) else None

            # remove from overview-files
            ms.modify_shape(path_exact_footprints_shape_invalid, image_id=image_id, modus="delete")
            ms.modify_shape(path_exact_photocenters_shape_invalid, image_id=image_id, modus="delete")
            ms.modify_shape(path_error_vector_shape_invalid, image_id=image_id, modus="delete")

    # Fifth, we check if the image did fail before
    if retry_failed is False and mc.modify_csv(path_failed_csv, image_id, "check"):
        p.print_v(f"{image_id} failed previously", verbose=verbose, pbar=pbar, color="red")
        return "failed"

    # remove from the list with failed and too few csv
    mc.modify_csv(path_failed_csv, image_id, "delete")
    mc.modify_csv(path_too_few_tps_csv, image_id, "delete")

    # remove the tiff from the geo-referenced folder
    os.remove(path_tiff) if os.path.exists(path_tiff) else None

    # remove the tie-point files from the geo-referenced folder
    ms.modify_shape(path_tie_points_shp, image_id=None, modus="hard_delete")
    os.remove(path_tie_points_img) if os.path.exists(path_tie_points_img) else None

    # remove from overview-files
    ms.modify_shape(path_exact_footprints_shape, image_id=image_id, modus="delete")
    ms.modify_shape(path_exact_photocenters_shape, image_id=image_id, modus="delete")
    ms.modify_shape(path_error_vector_shape, image_id=image_id, modus="delete")

    # remove from db (but only if not image before)
    if data['footprint'] is not None and previous_georef_type == "satellite" and save_in_db:
        sql_string = "UPDATE images_extracted SET footprint = NULL, " \
                     "position_exact = NULL, position_error_vector = NULL, footprint_type = NULL " \
                     f"WHERE image_id='{image_id}'"
        ctd.edit_data_in_db(sql_string, catch=catch, verbose=False, pbar=pbar)

    # load the image
    image_with_borders = liff.load_image_from_file(image_id, catch=catch,
                                                   verbose=False, pbar=pbar)

    # if something went wrong loading the image
    if image_with_borders is None:
        p.print_v(f"Error loading {image_id}", verbose=verbose, pbar=pbar, color="red")

        data_dict = {
            'reason': "loading_image"
        }

        # add to failed csv
        mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

        return "failed"

    # get the border dimensions
    image_no_borders, border_dims = rb.remove_borders(image_with_borders, image_id=image_id,
                                                      return_edge_dims=True,
                                                      catch=catch, verbose=False, pbar=pbar)

    # if something went wrong loading the borders
    if image_no_borders is None:
        p.print_v(f"Error removing borders from {image_id}", verbose=verbose, pbar=pbar, color="red")

        data_dict = {
            'reason': "removing_borders"
        }

        # add to failed csv
        mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

        return "failed"

    # rotate the image
    image = ri.rotate_image(image_no_borders, data["azimuth"], start_angle=0)

    # if we cannot rotate the image
    if image is None:
        p.print_v(f"Error rotating {image_id}", verbose=verbose, pbar=pbar, color="red")

        data_dict = {
            'reason': "rotating_image"
        }

        # add to failed csv
        mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

        return "failed"

    # no need to look for tie-points, if we already have them
    if debug_recalculate_tie_points is False and os.path.isfile(path_tie_points_shp):
        p.print_v(f"Tie-points for {image_id} already existing")

        try:
            # read the shapefile
            df = gpd.read_file(path_tie_points_shp)

            # get tps and conf
            tps_coords = df.iloc[:, :2]
            tps_img = df.iloc[:, 4:6]
            tps_abs = pd.concat([tps_coords, tps_img], axis=1).values
            conf = df.iloc[:, 6].values

        except (Exception,):
            tps_abs = None
    else:
        tps_abs = None

    # we need to look for tie-points
    if tps_abs is None:

        # we need the approx approx_footprint of image (three options available)
        # Option 1: we are manually submitting a approx_footprint:
        if existing_footprint is not None:
            footprint_approx = existing_footprint

        # Option 2: we use an existing good approx_footprint based on image calculations
        elif data['footprint'] is not None and data['footprint_type'] == "image":
            footprint_approx = shapely.wkt.loads(data['footprint'])

            p.print_v("Using exact approx_footprint from image", verbose=verbose, pbar=pbar)

        # Option 3: we already have an approx approx_footprint
        elif recalculate_approx_footprint is False and data['footprint_approx'] is not None:
            footprint_approx = shapely.wkt.loads(data['footprint_approx'])

            p.print_v("Using existing approx approx_footprint", verbose=verbose, pbar=pbar)

        # Option 4: we need to calculate it
        else:

            p.print_v("Calculate approx approx_footprint", verbose=verbose, pbar=pbar)

            # for the calculation of the approx approx_footprint we need some more information
            focal_length = data["focal_length"]
            height = data["height"]
            x = data["x_coords"]
            y = data["y_coords"]
            view_direction = data["view_direction"]
            angle = data["azimuth"]

            # we need an angle (and that we cannot substitute)
            if angle is None:
                p.print_v(f"{image_id} is missing an angle", verbose=verbose,
                          pbar=pbar, color="red")

                data_dict = {
                    'reason': "missing_angle"
                }

                # add to failed csv
                mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                return "failed"

            # we need a focal length
            if focal_length is None:
                # if we don't want to use avg values and don't have a focal length
                # -> this image we cannot use
                if debug_use_avg_values is False:
                    p.print_v(f"{image_id} is missing a focal length", verbose=verbose,
                              pbar=pbar, color="red")

                    data_dict = {
                        'reason': "missing_focal_length"
                    }

                    # add to failed csv
                    mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                    return "failed"

                # we use an average value
                else:
                    focal_length = 152.457

            # we need a height
            if height is None or np.isnan(height):
                # if we don't want to use avg values and don't have a height
                # -> this image we cannot use
                if debug_use_avg_values is False:
                    p.print_v(f"{image_id} is missing a height", verbose=verbose,
                              pbar=pbar, color="red")

                    data_dict = {
                        'reason': "missing_height"
                    }

                    # add to failed csv
                    mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                    return "failed"

                # we use an average value
                else:
                    height = 25000

            # calculate an approximate approx_footprint based on the information we have from the shapefile
            footprint_approx = cfa.calc_footprint_approx(x, y, angle, view_direction,
                                                         height, focal_length,
                                                         catch=True, verbose=False, pbar=pbar)

            # sometimes we cannot calculate the approx approx_footprint
            if footprint_approx is None:
                p.print_v(f"For {image_id} no approx approx_footprint could be calculated", verbose=verbose,
                          pbar=pbar, color="red")

                data_dict = {
                    'reason': "calculating_approx_footprint"
                }

                # add to failed csv
                mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                return "failed"

            # save the approx approx_footprint in a shapefile if we want
            if debug_save_approx_footprints_shape:

                # Create a GeoDataFrame with the approx_footprint
                footprint_gdf = gpd.GeoDataFrame(geometry=[footprint_approx])
                footprint_gdf.crs = "EPSG:3031"
                footprint_gdf['image_id'] = image_id
                footprint_gdf['complexity'] = data['complexity']

                # Create the shapefile if it doesn't exist
                if not os.path.exists(path_approx_footprints_shape):

                    footprint_gdf.to_file(path_approx_footprints_shape)

                else:
                    # Read the existing shapefile
                    existing_gdf = gpd.read_file(path_approx_footprints_shape)

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
                    existing_gdf.to_file(path_approx_footprints_shape)

                if save_in_db:
                    sql_string = f"UPDATE images_extracted SET " \
                                 f"footprint_approx=ST_GeomFromText('{footprint_approx}') " \
                                 f"WHERE image_id='{image_id}'"
                    success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=False, pbar=None)
                    if success is False:
                        p.print_v(f"Something went wrong saving the approx approx_footprint in db for {image_id}",
                                  verbose=verbose, pbar=pbar, color="red")

                        data_dict = {
                            'reason': "saving_approx_footprint"
                        }

                        # add to failed csv
                        mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                        return "failed"

            # save the approx photocenter in a shapefile if we want
            if debug_save_approx_photocenters_shape:

                # to save the photocenter we need to calculate it from the approx_footprint
                x, y = ccp.calc_camera_position(footprint_approx)

                # create a shapely point that we can save in a db
                photocenter_approx = Point(x, y)

                # Create a GeoDataFrame with the photocenter
                photocenter_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x], [y]))
                photocenter_gdf.crs = "EPSG:3031"
                photocenter_gdf['image_id'] = image_id

                # Create the shapefile if it doesn't exist
                if not os.path.exists(path_approx_photocenters_shape):
                    photocenter_gdf.to_file(path_approx_photocenters_shape)
                else:
                    # Read the existing shapefile
                    existing_gdf = gpd.read_file(path_approx_photocenters_shape)

                    # Check if the image_id is already in our existing photocenters
                    if image_id in existing_gdf['image_id'].values:
                        # Update the entry in existing_gdf
                        existing_gdf.loc[existing_gdf['image_id'] == image_id,
                                         "geometry"] = photocenter_gdf.iloc[0]['geometry']
                    else:
                        # Append the new photocenter to the existing GeoDataFrame
                        existing_gdf = pd.concat([existing_gdf, photocenter_gdf], ignore_index=True)

                    # Save the updated GeoDataFrame to the shapefile
                    existing_gdf.to_file(path_approx_photocenters_shape)

                if save_in_db:
                    sql_string = f"UPDATE images_extracted SET " \
                                 f"position_approx=ST_GeomFromText('{photocenter_approx}') " \
                                 f"WHERE image_id='{image_id}'"
                    success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=False, pbar=None)
                    if success is False:
                        p.print_v(f"Something went wrong saving the approx photocenter in db for {image_id}",
                                  verbose=verbose, pbar=pbar, color="red")

                        data_dict = {
                            'reason': "saving_approx_photocenter"
                        }

                        # add to failed csv
                        mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                        return "failed"

        # buffer the approx approx_footprint
        footprint_buffered = bf.buffer_footprint(footprint_approx, buffer_val=buffer_val,
                                                 catch=False, verbose=False, pbar=pbar)

        # get the initial satellite bounds from the buffered approx_footprint
        sat_bounds = list(footprint_buffered.bounds)
        sat_bounds = [int(x) for x in sat_bounds]

        # we can adapt the bounds of an approx approx_footprint with information
        # from other already georeferenced images
        # however, we shouldn't do it if we georeferenced images that already are georeferenced
        # from images or where we provide a approx_footprint
        if debug_adapt_sat_bounds_with_existing and data["footprint_type"] != "image" and \
                existing_footprint is not None:

            # init the error vectors
            x_vec, y_vec = 0, 0

            # get the flight details of this image
            flight = image_id[2:9]
            image_nr = int(image_id[-4:])

            # get the flight data from the database
            sql_string = "SELECT image_id, position_error_vector FROM images_extracted " \
                         f"WHERE SUBSTRING(image_id, 3, 7) = '{flight}' AND " \
                         f"footprint_type='satellite'"
            fl_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

            # drop rows without exact approx_footprint
            fl_data = fl_data.dropna()

            # we only can calculate an error vector if we have data
            if fl_data.shape[0] > 0:

                # get x and y as separate columns
                fl_data[['x', 'y']] = fl_data['position_error_vector'].str.split(';', expand=True)
                fl_data['x'] = fl_data['x'].astype(float)
                fl_data['y'] = fl_data['y'].astype(float)

                # calculate using closest (or both)
                if error_vector_calculation in ["closest", "both"]:

                    # which would be the closest numbers
                    image_nr_min = image_nr - error_vector_order
                    image_nr_max = image_nr + error_vector_order

                    # with all numbers in between
                    numbers_between = [num for num in range(image_nr_min, image_nr_max + 1)]

                    # add flight path to the numbers
                    subset_ids = ['CA' + flight + '{:04d}'.format(_id) for _id in numbers_between]

                    # now get the ref_data for all these numbers
                    subset_ref_data = fl_data[fl_data['image_id'].isin(subset_ids)]

                    # if the order is too small it can happen that we don't have any images
                    if subset_ref_data.shape[0] == 0:
                        x_vec = 0
                        y_vec = 0
                    else:
                        x_vec = round(subset_ref_data['x'].mean(), 4)
                        y_vec = round(subset_ref_data['y'].mean(), 4)

                # calculate using avg (or both, but only if we didn't already found an error vector)
                if error_vector_calculation in ["avg", "both"] and x_vec == 0 and y_vec == 0:
                    x_vec = round(fl_data['x'].mean(), 4)
                    y_vec = round(fl_data['y'].mean(), 4)

            sat_bounds[0] = sat_bounds[0] + x_vec
            sat_bounds[1] = sat_bounds[1] + y_vec
            sat_bounds[2] = sat_bounds[2] + x_vec
            sat_bounds[3] = sat_bounds[3] + y_vec

        # check if we want to use a certain month for loading a satellite image
        if debug_use_month:
            month = data["date_month"]
        else:
            month = 0

        # load the initial approx satellite image
        sat, sat_transform, lst_sat_images = lsd.load_satellite_data(sat_bounds,
                                                                     month=month,
                                                                     fallback_month=True,
                                                                     return_transform=True,
                                                                     return_used_images=True,
                                                                     catch=True, verbose=False,
                                                                     pbar=pbar)

        # if the satellite is none something went wrong
        if sat is None:
            p.print_v(f"{image_id} does not have a satellite image", verbose=verbose,
                      pbar=pbar, color="red")

            data_dict = {
                'reason': "no_satellite_image"
            }

            # add to failed csv
            mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

            return "failed"

        # rotate the original image (just for change_factor_rotated)
        img_rot = ri.rotate_image(image_no_borders, data['azimuth'], start_angle=0)

        # adjust the image so that pixel size is similar to satellite image
        image_with_borders_adjusted = air.adjust_image_resolution(sat,
                                                                  sat_bounds,
                                                                  image_with_borders,
                                                                  footprint_approx.bounds)

        # get the difference between the images with borders
        change_y_with_borders = image_with_borders.shape[0] / image_with_borders_adjusted.shape[0]
        change_x_with_borders = image_with_borders.shape[1] / image_with_borders_adjusted.shape[1]

        # adapt the border dimension to account for smaller image
        border_dims_adjusted = copy.deepcopy(border_dims)
        if change_y_with_borders > 0 or change_x_with_borders > 0:
            change_y_with_borders = 1 / change_y_with_borders
            change_x_with_borders = 1 / change_x_with_borders
        border_dims_adjusted[0] = int(border_dims[0] * change_x_with_borders)
        border_dims_adjusted[1] = int(border_dims[1] * change_x_with_borders)
        border_dims_adjusted[2] = int(border_dims[2] * change_y_with_borders)
        border_dims_adjusted[3] = int(border_dims[3] * change_y_with_borders)

        # remove the border from the small image and then rotate
        image_no_borders_adjusted = image_with_borders_adjusted[border_dims_adjusted[2]:border_dims_adjusted[3],
                                    border_dims_adjusted[0]:border_dims_adjusted[1]]

        print("image_no_borders_adjusted", image_no_borders_adjusted.shape)

        # get the difference between the images without borders
        change_y_no_borders = image_no_borders.shape[0] / image_no_borders_adjusted.shape[0]
        change_x_no_borders = image_no_borders.shape[1] / image_no_borders_adjusted.shape[1]
        if change_y_no_borders > 0 or change_x_no_borders > 0:
            change_y_no_borders = 1 / change_y_no_borders
            change_x_no_borders = 1 / change_x_no_borders

        # enhance images to improve tie-point detection
        if debug_enhance_images:
            # we need this copy later
            images_copy = copy.deepcopy(image_no_borders_adjusted)

            # the actual enhancement
            image_no_borders_adjusted = eh.enhance_image(image_no_borders_adjusted,
                                                         scale=(enhancement_min, enhancement_max))

            print("image_no_borders_adjusted2", image_no_borders_adjusted.shape)

        # rotate the images
        image_adjusted = ri.rotate_image(image_no_borders_adjusted, data['azimuth'], start_angle=0)

        # get change for rotated image
        change_y_rotated = img_rot.shape[0] / image_adjusted.shape[0]
        change_x_rotated = img_rot.shape[1] / image_adjusted.shape[1]
        if change_y_rotated > 0 or change_y_rotated > 0:
            change_y_rotated = 1 / change_y_rotated
            change_x_rotated = 1 / change_x_rotated

        print("image_adjusted", image_adjusted.shape)

        # should we show the initial images
        if debug_show_initial_images:
            di.display_images([sat, image_adjusted], title=f"{image_id}")

        p.print_v(f"Calculate tie-points for {image_id}", verbose=verbose, pbar=pbar)

        # first check of how many tps we get without threshold or filtering
        try:
            tps, conf = ftp.find_tie_points(sat, image_adjusted,
                                            min_threshold=0,
                                            filter_outliers=False,
                                            additional_matching=True, extra_matching=True,
                                            catch=True, verbose=False, pbar=None)
        except (Exception,) as e:
            if catch:
                p.print_v(f"Error in calculating tie-points for {image_id}",
                          verbose=verbose, pbar=pbar, color="red")

                data_dict = {
                    'reason': "calculating_tps"
                }

                # add to failed csv
                mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                return "failed"
            else:
                raise e

        # get number of tie points
        if tps is None:
            nr_points = 0
        else:
            nr_points = tps.shape[0]

        # get average conf
        if conf is None:
            avg_conf = 0
        else:
            avg_conf = np.mean(np.asarray(conf))
            if np.isnan(avg_conf):
                avg_conf = 0

        p.print_v(f"{nr_points} initial tie-points are found for {image_id}",
                  verbose=verbose, pbar=pbar)

        # show the initial tie-points
        if nr_points > 0 and debug_show_tie_points_initial:
            dt.display_tiepoints([sat, image_adjusted], tps, conf, verbose=False,
                                 title=f"Tie-points for {image_id} ({tps.shape[0]})")

        # check if we need to look further
        if locate_image and nr_points <= min_nr_of_tps_final and debug_located_coords is None:

            # get the width and height of the satellite image
            sat_width = sat.shape[1] * np.abs(sat_transform[0])
            sat_height = sat.shape[2] * np.abs(sat_transform[4])

            # calculate the step size
            step_x = int((1 - location_overlap) * sat_width)
            step_y = int((1 - location_overlap) * sat_height)

            # save which combinations did we already check (e. g the center like here)
            lst_checked_combinations = [[0, 0]]

            # save the best values (start with initial values from center)
            best_tps = tps
            best_conf = conf
            best_combination = [0, 0]
            best_nr_of_points = nr_points
            best_avg_conf = avg_conf

            # best satellite
            best_sat = copy.deepcopy(sat)
            best_sat_transform = copy.deepcopy(sat_transform)
            best_sat_bounds = copy.deepcopy(sat_bounds)

            # iterate the images around our image (ignoring the center)
            for order in range(1, location_max_order):

                # create a combinations dict
                combinations = []
                for i in range(-order * step_x, (order + 1) * step_x, step_x):
                    for j in range(-order * step_y, (order + 1) * step_y, step_y):
                        combinations.append([i, j])

                # check all combinations
                for combination in combinations:

                    # check if we already checked this combination
                    if combination in lst_checked_combinations:
                        continue

                    # add the combination to the list of checked combinations
                    lst_checked_combinations.append(combination)

                    p.print_v(f"Check combination {combination} for {image_id}, (Order {order})",
                              verbose=verbose, pbar=pbar)

                    # adapt the satellite bounds for this combination
                    sat_bounds_combi = copy.deepcopy(sat_bounds)

                    # first adapt x
                    sat_bounds_combi[0] = sat_bounds_combi[0] + combination[0]
                    sat_bounds_combi[2] = sat_bounds_combi[2] + combination[0]

                    # and then y
                    sat_bounds_combi[1] = sat_bounds_combi[1] + combination[1]
                    sat_bounds_combi[3] = sat_bounds_combi[3] + combination[1]

                    # ignore mean of empty slice & invalid divide value warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        # get the satellite image
                        sat_combi, sat_transform_combi, lst_sat_images = lsd.load_satellite_data(sat_bounds_combi,
                                                                                                 return_transform=True,
                                                                                                 return_used_images=True,
                                                                                                 catch=True,
                                                                                                 verbose=False,
                                                                                                 pbar=None)

                        if sat_combi is None:
                            p.print_v("No satellite image could be found",
                                      verbose=verbose, pbar=pbar, color="red")
                        else:

                            # get tie-points between satellite image and historical image
                            tps_combi, conf_combi = ftp.find_tie_points(sat_combi, image_adjusted,
                                                                        min_threshold=0,
                                                                        filter_outliers=False,
                                                                        additional_matching=True,
                                                                        extra_matching=True,
                                                                        catch=True, verbose=False, pbar=None)

                            # get number of tie points
                            if tps_combi is None:
                                nr_points_combi = 0
                            else:
                                nr_points_combi = tps_combi.shape[0]

                            # get average conf
                            if conf_combi is None:
                                avg_conf_combi = 0
                            else:
                                avg_conf_combi = np.mean(np.asarray(conf_combi))
                                if np.isnan(avg_conf_combi):
                                    avg_conf_combi = 0

                    if sat_combi is not None:
                        p.print_v(f"{nr_points_combi} tie-points ({avg_conf_combi}) found at {combination}",
                                  verbose=verbose, pbar=pbar)

                        # check if we have the best combination
                        if location_success_criteria == "points":

                            if nr_points_combi > best_nr_of_points or \
                                    (nr_points_combi == best_nr_of_points and
                                     avg_conf_combi > best_avg_conf):
                                best_tps = copy.deepcopy(tps_combi)
                                best_conf = copy.deepcopy(conf_combi)

                                best_combination = copy.deepcopy(combination)
                                best_nr_of_points = copy.deepcopy(nr_points_combi)
                                best_avg_conf = copy.deepcopy(avg_conf_combi)

                                best_sat = copy.deepcopy(sat_combi)
                                best_sat_transform = copy.deepcopy(sat_transform_combi)
                                best_sat_bounds = copy.deepcopy(sat_bounds_combi)

                        elif location_success_criteria == "conf":

                            if avg_conf_combi > best_avg_conf or \
                                    (avg_conf_combi == best_avg_conf and
                                     nr_points_combi > best_nr_of_points):
                                best_tps = copy.deepcopy(tps_combi)
                                best_conf = copy.deepcopy(conf_combi)

                                best_combination = copy.deepcopy(combination)
                                best_nr_of_points = copy.deepcopy(nr_points_combi)
                                best_avg_conf = copy.deepcopy(avg_conf_combi)

                                best_sat = copy.deepcopy(sat_combi)
                                best_sat_transform = copy.deepcopy(sat_transform_combi)
                                best_sat_bounds = copy.deepcopy(sat_bounds_combi)

                        if best_nr_of_points > max_nr_stop_combination:
                            p.print_v(f"{best_nr_of_points} found in an combination -> stop searching!",
                                      verbose=verbose, pbar=pbar)
                            break

                # do we need to break because of sat_combi being None
                if sat_combi is None:
                    break

                # do we already have a nice overlap -> break the loop
                if location_success_criteria == "points" and best_nr_of_points > min_nr_of_tps_final:
                    break
                elif location_success_criteria == "conf" and best_avg_conf > min_threshold:
                    break

            # continue to the next image
            if sat_combi is None:
                p.print_v("The satellite image couldn't be loaded", verbose=verbose,
                          color="red", pbar=pbar)

                data_dict = {
                    'reason': "loading_satellite_image"
                }

                # add to failed csv
                mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                return "failed"

            p.print_v(f"Best combination is {best_combination} with "  # noqa
                      f"{best_nr_of_points} tie-points ({best_avg_conf})",
                      verbose=verbose, pbar=pbar)  # noqa

            if best_tps is None:
                p.print_v("Failed with best tps", verbose=verbose,
                          color="red", pbar=pbar)

                data_dict = {
                    'reason': "best_tps_is_none"
                }

                # add to failed csv
                mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                return "failed"

            if best_tps.shape[0] > 0 and debug_show_tie_points_located:
                dt.display_tiepoints([sat, image_adjusted], best_tps, best_conf, verbose=False,
                                     title=f"Tie-points for {image_id} ({best_tps.shape[0]})")

            # extract best values
            tps = copy.deepcopy(best_tps)
            conf = copy.deepcopy(best_conf)
            sat = copy.deepcopy(best_sat)
            sat_transform = copy.deepcopy(best_sat_transform)
            sat_bounds = copy.deepcopy(best_sat_bounds)

        # we've determined the best combination already manually, so we just can load it
        elif debug_located_coords is not None:

            sat_bounds = sat_bounds
            # first adapt x
            sat_bounds[0] = sat_bounds[0] + debug_located_coords[0]
            sat_bounds[2] = sat_bounds[2] + debug_located_coords[0]

            # and then y
            sat_bounds[1] = sat_bounds[1] + debug_located_coords[1]
            sat_bounds[3] = sat_bounds[3] + debug_located_coords[1]

            # get the satellite image
            sat, sat_transform, lst_sat_images = lsd.load_satellite_data(sat_bounds,
                                                                        return_transform=True,
                                                                        return_used_images=True,
                                                                        catch=True,
                                                                        verbose=False,
                                                                        pbar=None)

            # get tie-points between satellite image and historical image
            tps, conf = ftp.find_tie_points(sat, image_adjusted,
                                            min_threshold=0,
                                            filter_outliers=False,
                                            additional_matching=True,
                                            extra_matching=True,
                                            catch=True, verbose=False, pbar=None)

            p.print_v(f"Best combination is {debug_located_coords} with "  # noqa
                      f"{tps.shape[0]} tie-points (manually located)",
                      verbose=verbose, pbar=pbar)  # noqa

        # check number of tps
        if tps.shape[0] < min_nr_of_tps_running:
            p.print_v(f"{image_id} has too few tie-points for further search",
                      color="red", verbose=verbose, pbar=pbar)

            data_dict = {
                'tie_points': tps.shape[0]
            }

            # add to too few tps csv
            mc.modify_csv(path_too_few_tps_csv, image_id, "add", data=data_dict)

            # remove the tie-point file
            # we need to make the path shorter, as we cannot move only the shp file but others as well
            short_path = path_tie_points_shp[:-3]
            short_path_too_few_tps = path_tie_points_shp_too_few_tps[:-3]

            # relocate all parts of the tie point shape to the invalid folder
            os.rename(short_path + "cpg", short_path_too_few_tps + "cpg") if os.path.exists(short_path + "cpg") else None
            os.rename(short_path + "dbf", short_path_too_few_tps + "dbf") if os.path.exists(short_path + "dbf") else None
            os.rename(short_path + "prj", short_path_too_few_tps + "prj") if os.path.exists(short_path + "prj") else None
            os.rename(short_path + "shp", short_path_too_few_tps + "shp") if os.path.exists(short_path + "shp") else None
            os.rename(short_path + "shx", short_path_too_few_tps + "shx") if os.path.exists(short_path + "shx") else None

            # relocate image
            os.rename(path_tie_points_img, path_tie_points_img_too_few_tps) if os.path.exists(path_tie_points_img) else None

            return "too_few_tps"

        # tweak image if wished
        if debug_tweak_image:
            p.print_v("Start tweaking images", verbose=verbose, pbar=pbar)

            # get the start values
            sat_bounds_tweaked = copy.deepcopy(sat_bounds)

            # init the best values
            sat_tweaked_best = copy.deepcopy(sat)
            sat_transform_tweaked_best = copy.deepcopy(sat_transform)
            sat_bounds_tweaked_best = copy.deepcopy(sat_bounds)
            tweaked_best_nr_points = copy.deepcopy(tps.shape[0])

            tps_tweaked_best = copy.deepcopy(tps)
            conf_tweaked_best = copy.deepcopy(conf)

            # how often did we tweaked the image
            counter = 0

            # tweak the image in a loop
            while True:

                # we only want to tweak the image a maximum number of times
                counter = counter + 1
                if counter == tweaking_max_iterations:
                    break

                # find the direction in which we need to change the satellite image
                step_y, step_x = ffd.find_footprint_direction(image_adjusted, tps[:, 2:4], conf)

                p.print_v(f"Tweak image ({counter}/{tweaking_max_iterations}) with ({step_y}, {step_x})",
                          verbose=verbose, pbar=pbar)

                # tweak the satellite bounds
                sat_bounds_tweaked[0] = sat_bounds_tweaked[0] + step_x
                sat_bounds_tweaked[1] = sat_bounds_tweaked[1] + step_y
                sat_bounds_tweaked[2] = sat_bounds_tweaked[2] + step_x
                sat_bounds_tweaked[3] = sat_bounds_tweaked[3] + step_y

                # get the tweaked satellite image
                sat_tweaked, sat_transform_tweaked, lst_sat_images = lsd.load_satellite_data(sat_bounds_tweaked,
                                                                                             month=month,
                                                                                             fallback_month=True,
                                                                                             return_transform=True,
                                                                                             return_used_images=True,
                                                                                             catch=True,
                                                                                             verbose=False,
                                                                                             pbar=pbar)

                # catch if something went wrong loading the satellite image
                if sat_tweaked is None:
                    p.print_v("Failed: load tweaked satellite image", verbose=verbose, pbar=pbar, color="red")
                    break

                # get tie-points between satellite image and historical image
                tps_tweaked, conf_tweaked = ftp.find_tie_points(sat_tweaked, image_adjusted,
                                                                min_threshold=0,
                                                                filter_outliers=False,
                                                                additional_matching=True, extra_matching=True,
                                                                catch=False, verbose=False, pbar=None)
                if tps_tweaked is None:
                    p.print_v("Failed: finding tweaked tps", verbose=verbose, pbar=pbar, color="red")

                    # remove the tie-point file
                    ms.modify_shape(path_tie_points_shp, image_id=None, modus="hard_delete")
                    os.remove(path_tie_points_img) if os.path.exists(path_tie_points_img) else None

                    data_dict = {
                        'reason': "tweaking_tps"
                    }

                    # add to failed csv
                    mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

                    return "failed"

                p.print_v(f"{tps_tweaked.shape[0]} points found in tweak "
                          f"{counter} of {tweaking_max_iterations}", verbose=verbose, pbar=pbar)

                # if the number of points is going up we need to save the params
                if tps_tweaked.shape[0] >= tweaked_best_nr_points:
                    # save the best params
                    tweaked_best_nr_points = tps_tweaked.shape[0]
                    sat_tweaked_best = copy.deepcopy(sat_tweaked)
                    sat_transform_tweaked_best = copy.deepcopy(sat_transform_tweaked)
                    sat_bounds_tweaked_best = copy.deepcopy(sat_bounds_tweaked)

                    tps_tweaked_best = copy.deepcopy(tps_tweaked)
                    conf_tweaked_best = copy.deepcopy(conf_tweaked)

                else:
                    p.print_v(f"Points going down ({tps.shape[0]} < {tweaked_best_nr_points})",
                              verbose=verbose, pbar=pbar)
                    # stop the loop
                    break

            # restore the best settings again
            tps = copy.deepcopy(tps_tweaked_best)  # noqa
            conf = copy.deepcopy(conf_tweaked_best)  # noqa
            sat = copy.deepcopy(sat_tweaked_best)
            sat_transform = copy.deepcopy(sat_transform_tweaked_best)
            sat_bounds = copy.deepcopy(sat_bounds_tweaked_best)  # noqa

            if tps.shape[0] < min_nr_of_tps_running:
                p.print_v(f"Too few tie-points ({tps.shape[0]}) found for {image_id}",
                          verbose=verbose, pbar=pbar, color="red")

                data_dict = {
                    'tie_points': tps.shape[0]
                }

                mc.modify_csv(path_too_few_tps_csv, image_id, "add", data=data_dict)

                # remove the tie-point file
                # we need to make the path shorter, as we cannot move only the shp file but others as well
                short_path = path_tie_points_shp[:-3]
                short_path_too_few_tps = path_tie_points_shp_too_few_tps[:-3]

                # relocate all parts of the tie point shape to the invalid folder
                os.rename(short_path + "cpg", short_path_too_few_tps + "cpg") if os.path.exists(
                    short_path + "cpg") else None
                os.rename(short_path + "dbf", short_path_too_few_tps + "dbf") if os.path.exists(
                    short_path + "dbf") else None
                os.rename(short_path + "prj", short_path_too_few_tps + "prj") if os.path.exists(
                    short_path + "prj") else None
                os.rename(short_path + "shp", short_path_too_few_tps + "shp") if os.path.exists(
                    short_path + "shp") else None
                os.rename(short_path + "shx", short_path_too_few_tps + "shx") if os.path.exists(
                    short_path + "shx") else None

                # relocate image
                os.rename(path_tie_points_img, path_tie_points_img_too_few_tps) if os.path.exists(
                    path_tie_points_img) else None

                return "too_few_tps"

            else:
                p.print_v(f"Tweaking for {image_id} finished with {tps.shape[0]} tie-points",
                          verbose=verbose, pbar=pbar)

        if debug_print_used_satellite_img:
            p.print_v(f"Following satellite images are used: {lst_sat_images}")

        if tps.shape[0] > 0 and debug_show_tie_points_final_unfiltered:
            dt.display_tiepoints([sat, image_adjusted], tps, conf, verbose=False,
                                 title=f"Tie-points for {image_id} ({tps.shape[0]})")

        # filter with ransac and angle
        if filter_outliers:
            tps, conf = fto.filter_tp_outliers(tps, conf, use_angle=True, angle_threshold=10,
                                               verbose=verbose, pbar=pbar)

        if tps.shape[0] > 0 and debug_show_tie_points_final:
            print("TEEST", image_adjusted.shape)
            dt.display_tiepoints([sat, image_adjusted], tps, conf, verbose=False,
                                 title=f"Final Tie-points for {image_id} ({tps.shape[0]})")

        # intermediate check of the image
        if tps.shape[0] < min_nr_of_tps_running:
            p.print_v(f"Too few tie-points ({tps.shape[0]}) to georeference {image_id}",
                      verbose=verbose, pbar=pbar, color="red")

            data_dict = {
                'tie_points': tps.shape[0]
            }

            mc.modify_csv(path_too_few_tps_csv, image_id, "add", data=data_dict)

            # remove the tie-point file
            # we need to make the path shorter, as we cannot move only the shp file but others as well
            short_path = path_tie_points_shp[:-3]
            short_path_too_few_tps = path_tie_points_shp_too_few_tps[:-3]

            # relocate all parts of the tie point shape to the invalid folder
            os.rename(short_path + "cpg", short_path_too_few_tps + "cpg") if os.path.exists(
                short_path + "cpg") else None
            os.rename(short_path + "dbf", short_path_too_few_tps + "dbf") if os.path.exists(
                short_path + "dbf") else None
            os.rename(short_path + "prj", short_path_too_few_tps + "prj") if os.path.exists(
                short_path + "prj") else None
            os.rename(short_path + "shp", short_path_too_few_tps + "shp") if os.path.exists(
                short_path + "shp") else None
            os.rename(short_path + "shx", short_path_too_few_tps + "shx") if os.path.exists(
                short_path + "shx") else None

            # relocate image
            os.rename(path_tie_points_img, path_tie_points_img_too_few_tps) if os.path.exists(
                path_tie_points_img) else None

            return "too_few_tps"

        # currently the images are adjusted for satellite resolution -> they're smaller
        # if this is false, we enlarge them again
        if debug_use_small_image_size is False:

            # get the image with original image size
            image = copy.deepcopy(image_no_borders)

            # we still need to rotate it
            image = ri.rotate_image(image, data['azimuth'], start_angle=0)

            # adapt the tps to account for the original image-size
            print(np.amax(tps[:,2]), np.amax(tps[:,3]))
            tps[:, 2] = (tps[:, 2] * (1 / change_x_rotated)).astype(int)
            tps[:, 3] = (tps[:, 3] * (1 / change_y_rotated)).astype(int)

            print(change_x_rotated, change_y_rotated)
            print(change_x_no_borders, change_x_with_borders)
            print(change_y_no_borders, change_y_with_borders)
            print(image.shape)
            print(np.amax(tps[:,2]), np.amax(tps[:,3]))
        else:
            if debug_enhance_images:
                image = images_copy  # noqa
            else:
                image = image_adjusted

        # save the tie-points as images
        if debug_save_tie_points_image:
            p.print_v(f"Tie-points image saved at {path_tie_points_img}", verbose=verbose, pbar=pbar)
            dt.display_tiepoints([sat, image], points=tps, confidences=conf,
                                 title=f"Tie points for {image_id} ({tps.shape[0]})",
                                 save_path=path_tie_points_img)

        # copy the tie-points for absolute coords
        tps_abs = copy.deepcopy(tps)

        # project satellite points to absolute coords
        tps_abs[:, 0] = tps_abs[:, 0] * sat_transform[0]
        tps_abs[:, 1] = tps_abs[:, 1] * sat_transform[4]
        tps_abs[:, 0] = tps_abs[:, 0] + sat_transform[2]
        tps_abs[:, 1] = sat_transform[5] - sat.shape[0] * sat_transform[
            4] + tps_abs[:, 1]

        # save the tie-points as shapes
        if debug_save_tie_points_shape:
            # Create a DataFrame from the NumPy array
            df = pd.DataFrame({
                'x': tps_abs[:, 0], 'y': tps_abs[:, 1],
                'x_sat': tps[:, 0], 'y_sat': tps[:, 1],
                'x_pic': tps[:, 2], 'y_pic': tps[:, 3],
                'conf': conf,
                'x_fac_bor': change_x_with_borders, 'y_fac_bor': change_y_with_borders,
                'x_fac_nor': change_x_no_borders, 'y_fac_nor': change_y_no_borders,
                'x_fac_rot': change_x_rotated, 'y_fac_rot': change_y_rotated,
                'sat_min_x': sat_bounds[0], 'sat_max_x': sat_bounds[2],
                'sat_min_y': sat_bounds[1], 'sat_max_y': sat_bounds[3]
            })

            # convert to geopandas
            geometry = gpd.points_from_xy(x=df['x'], y=df['y'])
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:3031')

            # save to file
            gdf.to_file(path_tie_points_shp, driver='ESRI Shapefile')

    # final check of the tie-points
    if tps_abs.shape[0] < min_nr_of_tps_final:
        p.print_v(f"Too few tie-points ({tps_abs.shape[0]}) to georeference {image_id}",
                  verbose=verbose, pbar=pbar, color="red")

        data_dict = {
            'tie_points': tps_abs.shape[0]
        }

        mc.modify_csv(path_too_few_tps_csv, image_id, "add", data=data_dict)

        # remove the tie-point file
        # we need to make the path shorter, as we cannot move only the shp file but others as well
        short_path = path_tie_points_shp[:-3]
        short_path_too_few_tps = path_tie_points_shp_too_few_tps[:-3]

        # relocate all parts of the tie point shape to the invalid folder
        os.rename(short_path + "cpg", short_path_too_few_tps + "cpg") if os.path.exists(short_path + "cpg") else None
        os.rename(short_path + "dbf", short_path_too_few_tps + "dbf") if os.path.exists(short_path + "dbf") else None
        os.rename(short_path + "prj", short_path_too_few_tps + "prj") if os.path.exists(short_path + "prj") else None
        os.rename(short_path + "shp", short_path_too_few_tps + "shp") if os.path.exists(short_path + "shp") else None
        os.rename(short_path + "shx", short_path_too_few_tps + "shx") if os.path.exists(short_path + "shx") else None

        # relocate image
        os.rename(path_tie_points_img, path_tie_points_img_too_few_tps) if os.path.exists(path_tie_points_img) else None

        return "too_few_tps"

    # get avg conf of the tps
    avg_conf = np.mean(np.asarray(conf))

    # the actual geo-referencing
    if debug_georeference:

        # get the transform
        transform = ag.apply_gcps(path_tiff, image, tps_abs,
                                  transform_method, transform_order, catch=_catch)

        if transform is None:
            p.print_v(f"Something went wrong applying gcps to {image_id}", color="red",
                      verbose=verbose, pbar=pbar)

            data_dict = {
                'reason': "applying_gcps"
            }

            # add to failed csv
            mc.modify_csv(path_failed_csv, image_id, "add", data=data_dict)

            return "failed"

        # check if the image was geo-referenced correctly
        if debug_check_georef:
            image_is_valid = cgi.check_georef_image(path_tiff,
                                                    catch=catch, verbose=verbose, pbar=pbar)

        # without checks images are always valid
        else:
            image_is_valid = True

        # calculate residuals
        debug_calc_residuals = False
        if debug_calc_residuals:
            x_image, y_image, x_known, y_known = tps_abs[:, 2], tps_abs[:, 3], tps_abs[:, 0], tps_abs[:, 1]

            a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f

            # Applying the affine transformation
            x_transformed = a * x_image + b * y_image + c
            y_transformed = d * x_image + e * y_image + f

            # Calculating the residual errors
            errors = np.sqrt((x_transformed - x_known) ** 2 + (y_transformed - y_known) ** 2)

            import matplotlib.pyplot as plt
            try:
                plt.close()
            except (Exception,):
                pass

            plt.figure()
            plt.imshow(image)
            # Scatter plot of tie-points with color indicating the magnitude of the residual
            sc = plt.scatter(tps_abs[:, 2], tps_abs[:, 3], c=errors,
                             cmap='viridis', s=100, edgecolor='k')
            plt.colorbar(sc, label="Residual Error")
            plt.title("Residual Errors of Tie-Points")
            plt.show()

        # get the exact approx_footprint
        footprint_exact = citf.convert_image_to_footprint(image, image_id, transform,
                                                          catch=catch, verbose=False, pbar=pbar)

        # to save the photocenter we need to calculate it from the approx_footprint
        x, y = ccp.calc_camera_position(footprint_exact)

        # create a shapely point that we can save in a db
        photocenter_exact = Point(x, y)

        # calculate the approx photocenter
        approx_x, approx_y = ccp.calc_camera_position(footprint_approx)

        # create error vector
        error_vector = LineString([(x, y), (approx_x, approx_y)])

        # create error vector for db
        error_x = x - approx_x
        error_y = y - approx_y

        # for valid images we can create a approx_footprint
        if image_is_valid:

            p.print_v(f"{image_id} successfully georeferenced",
                      verbose=verbose, pbar=pbar, color="green")

            if previous_georef_type == "image":
                replacement_string = "img"
            elif previous_georef_type == "calc":
                replacement_string = "calc"

            path_prev_tiff = path_tiff.replace('sat', replacement_string)
            path_prev_tiff_invalid = path_tiff_invalid.replace('sat', replacement_string)

            # remove the tiff from the geo-referenced folder
            os.remove(path_prev_tiff) if os.path.exists(path_prev_tiff) else None
            os.remove(path_prev_tiff_invalid) if os.path.exists(path_prev_tiff_invalid) else None

            # remove the tie-point files from the geo-referenced folder
            ms.modify_shape(path_tie_points_shp, image_id=None, modus="hard_delete")
            os.remove(path_tie_points_img) if os.path.exists(path_tie_points_img) else None

            # remove from overview-files
            ms.modify_shape(path_exact_footprints_shape, image_id=image_id, modus="delete")
            ms.modify_shape(path_exact_photocenters_shape, image_id=image_id, modus="delete")
            ms.modify_shape(path_error_vector_shape, image_id=image_id, modus="delete")

            # save values in db
            if save_in_db:
                sql_string = f"UPDATE images_extracted SET " \
                             f"footprint=ST_GeomFromText('{footprint_exact.wkt}'), " \
                             f"position_exact=ST_GeomFromText('{photocenter_exact.wkt}'), " \
                             f"position_error_vector='{error_x};{error_y}', " \
                             f"footprint_type='satellite' " \
                             f"WHERE image_id='{image_id}'"
                success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=False, pbar=None)
                if success is False:
                    p.print_v(f"Something went wrong saving the exact approx_footprint in db for {image_id}",
                              verbose=verbose, pbar=pbar, color="red")
                    print("TODO REMOVE FROM FOLDERS")
                    exit()
                    return "failed"

            # save the exact approx_footprint in a shapefile if we want
            if debug_save_exact_footprints_shape:
                # get wkt from polygon
                poly_wkt = footprint_exact.wkt

                # create optional data
                footprint_data = {
                    'complexity': data['complexity'],
                    'nr_points': tps.shape[0],
                    'conf': avg_conf,
                    'sat_min_x': sat_bounds[0],
                    'sat_max_x': sat_bounds[2],
                    'sat_min_y': sat_bounds[1],
                    'sat_max_y': sat_bounds[3]
                }

                # add ids to the good overview shape files
                ms.modify_shape(path_exact_footprints_shape, image_id, "add",
                                geometry_data=poly_wkt, optional_data=footprint_data)

            # save the exact photocenter in a shapefile if we want
            if debug_save_exact_photocenters_shape:
                # get wkt from point
                point_wkt = photocenter_exact.wkt

                # add ids to the good overview shape files
                ms.modify_shape(path_exact_photocenters_shape, image_id, "add",
                                geometry_data=point_wkt)

            # save an error vector in a shapefile if we want
            if debug_save_error_vectors_shape:
                # get wkt from polygon
                line_wkt = error_vector.wkt

                # create optional data
                line_data = {
                    'error_x': error_x,
                    'error_y': error_y,
                }

                # add ids to the good overview shape files
                ms.modify_shape(path_error_vector_shape, image_id, "add",
                                geometry_data=line_wkt, optional_data=line_data)

            return "georeferenced"

        # invalid images we need to relocate and/or delete
        else:

            p.print_v(f"{image_id} is invalid", verbose=verbose, pbar=pbar, color="red")

            # relocate images to the invalid folder
            os.rename(path_tiff, path_tiff_invalid) if os.path.exists(path_tiff) else None

            # relocate tie point images to the invalid folder
            os.rename(path_tie_points_img, path_tie_points_img_invalid) if os.path.exists(path_tie_points_img) \
                else None

            # we need to make the path shorter, as we cannot move only the shp file but others as well
            short_path = path_tie_points_shp[:-3]
            short_path_invalid = path_tie_points_shp_invalid[:-3]

            # relocate all parts of the tie point shape to the invalid folder
            os.rename(short_path + "cpg", short_path_invalid + "cpg") if os.path.exists(short_path + "cpg") else None
            os.rename(short_path + "dbf", short_path_invalid + "dbf") if os.path.exists(short_path + "dbf") else None
            os.rename(short_path + "prj", short_path_invalid + "prj") if os.path.exists(short_path + "prj") else None
            os.rename(short_path + "shp", short_path_invalid + "shp") if os.path.exists(short_path + "shp") else None
            os.rename(short_path + "shx", short_path_invalid + "shx") if os.path.exists(short_path + "shx") else None

            # add ids to the invalid overview shape files
            # save the exact approx_footprint in a shapefile if we want
            if debug_save_exact_footprints_shape:
                # get wkt from polygon
                poly_wkt = footprint_exact.wkt

                # create optional data
                footprint_data = {
                    'complexity': data['complexity'],
                    'nr_points': tps.shape[0],
                    'conf': avg_conf,
                    'sat_min_x': sat_bounds[0],
                    'sat_max_x': sat_bounds[2],
                    'sat_min_y': sat_bounds[1],
                    'sat_max_y': sat_bounds[3]
                }

                # add ids to the invalid overview shape files
                ms.modify_shape(path_exact_footprints_shape_invalid, image_id, "add",
                                geometry_data=poly_wkt, optional_data=footprint_data)

            # save the exact photocenter in a shapefile if we want
            if debug_save_exact_photocenters_shape:
                # get wkt from point
                point_wkt = photocenter_exact.wkt

                # add ids to the invalid overview shape files
                ms.modify_shape(path_exact_photocenters_shape_invalid, image_id, "add",
                                geometry_data=point_wkt)

            # save an error vector in a shapefile if we want
            if debug_save_error_vectors_shape:
                # get wkt from polygon
                line_wkt = error_vector.wkt

                # create optional data
                line_data = {
                    'error_x': error_x,
                    'error_y': error_y,
                }

                # add ids to the invalid overview shape files
                ms.modify_shape(path_error_vector_shape_invalid, image_id, "add",
                                geometry_data=line_wkt, optional_data=line_data)

            return "invalid"

    raise ValueError("SHOULDN'T happen")


if __name__ == "__main__":
    __id = "CA181332V0082"

    status = georef_sat(__id, retry_invalid=True, retry_too_few_tps=True,
                        recalculate_approx_footprint=True,
                        save_in_db=True,
                        overwrite=True, catch=_catch, verbose=_verbose)
    print(status)
