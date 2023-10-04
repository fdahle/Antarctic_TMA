import copy
import cv2
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import warnings

import shapely.wkt
from tqdm import tqdm

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.rotate_image as ri

import display.display_images as di
import display.display_tiepoints as dt

import image_tie_points.find_tie_points as ftp

import image_georeferencing.sub.adjust_image_resolution as air
import image_georeferencing.sub.buffer_footprint as bf
import image_georeferencing.sub.calc_camera_position as ccp
import image_georeferencing.sub.calc_footprint_approx as cfa
import image_georeferencing.sub.load_satellite_data as lsd

import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.check_georef_image as cgi
import image_georeferencing.sub.find_footprint_direction as ffd
import playground.georeference.save_footprint as sf

_overwrite = False
_catch = False
_verbose = True

# general params
buffer_val = 2000
min_complexity = 0.05
min_nr_of_points = 5

debug_print_all = True
debug_use_small_image_size = False

print("ENABLE MULTICORES FOR GEO_REFERENCING")

# pre-selection of images
debug_only_image_footprint = True
debug_only_existing_tps = False  # If true, we only georeference files that already have tps as shapefiles

# settings to enable/disable some parts of the code
debug_use_existing_information = True
debug_locate_image = True
debug_tweak_image = True
debug_filter_outliers = True
debug_georeference = True
debug_final_check = True

# params for the approx footprints
use_avg_values = True  # if true, missing height or focal length are replaced by avg values
force_month = True  # if true, we are using not satellite images per month, but over all month

# set the params for debug_use_existing_information
error_vector_calculation = "both"  # can be 'closest' or 'avg' or 'both'
error_vector_order = 3  # how many of the closest error vectors should be taken?

# set the params for debug_locate_image
overlap = 1 / 3  # how much overlap do we want to have
max_order = 3  # what is the maximum order we're checking
success_criteria = "num_points"  # What is a success? 'num_points' or 'avg_conf'

# set the params for debug_tweak_image
max_tweaks = 10

# set the params for the gcp-creation
transform_method = "rasterio"  # can be 'gdal' or 'rasterio'
gdal_order = 1

# debug params for save
debug_save_approx_footprints_shape = True  # save the approx footprints in a shapefile
debug_save_approx_footprints_db = True # save the approx footprints in a database
debug_save_approx_photocenter_shape = True  # save the approx photocenters in a shapefile
debug_save_tps_shape = False
debug_save_tps_images = False
debug_save_exact_in_database = True

# debug params for show
debug_show_images = False
debug_show_tps = False

# the base path for the saved data
base_path = "/data_1/ATM/data_1/playground/georef2"


def georef2(image_ids, overwrite=False, catch=True, verbose=False):

    print(f"georeference {len(image_ids)} images")

    # this variable will contain the number of images we successfully georeferenced
    nr_new_images = 0

    # we need to overwrite if we want to use the image footprints
    if debug_only_image_footprint:
        overwrite = True

    # here we want to store the created data
    path_georef_tiffs = base_path + "/tiffs"
    path_georef_overview = base_path + "/overview"
    path_georef_tie_points_img = base_path + "/tie_points/images"
    path_georef_tie_points_shp = base_path + "/tie_points/shapes"

    # iterate all images
    for image_id in (pbar := tqdm(image_ids)):

        if debug_print_all:
            pbar = None

        try:
            p.print_v(f"georeference {image_id}", verbose=verbose, pbar=pbar)

            # get information from image and image_extracted
            sql_string = f"SELECT * FROM images WHERE image_id='{image_id}'"
            data_images = ctd.get_data_from_db(sql_string, catch=False, verbose=verbose)

            sql_string = "SELECT image_id, height, focal_length, complexity, " \
                         "ST_AsText(footprint_approx) AS footprint_approx, " \
                         "ST_AsText(footprint) AS footprint, " \
                         "footprint_type AS footprint_type " \
                         f"FROM images_extracted WHERE image_id ='{image_id}'"
            data_extracted = ctd.get_data_from_db(sql_string, catch=False, verbose=verbose)

            if debug_only_image_footprint and data_extracted['footprint_type'].iloc[0] != "image":
                p.print_v(f"{image_id} has no footprint type image!", verbose=verbose, pbar=None, color="green")
                continue

            if overwrite is False and data_extracted['footprint'].iloc[0] is not None and \
                    data_extracted['footprint_type'].iloc[0] == "satellite":
                p.print_v(f"{image_id} is already georeferenced!", verbose=verbose, pbar=None, color="green")
                continue

            # merge the information
            data = pd.merge(data_images, data_extracted, on='image_id').iloc[0]

            # get some basic information
            angle = data["azimuth"]
            month = data["date_month"]

            # sometimes we don't want to use specific month
            if force_month:
                month = 0

            # if we already have tie-points, we don't need to find them! (saves a lot of time)
            path_shape_file = base_path + "/shape_files/tie_points/" + image_id + "_points.shp"
            if overwrite is False and os.path.exists(path_shape_file):

                p.print_v(f"Use existing tie-points for {image_id}", verbose=verbose, pbar=pbar)

                # read the shapefile
                df = gpd.read_file(path_shape_file)

                # get tps and conf
                tps = df.iloc[2, :6].values
                conf = df.iloc[:, 7].values

                sat_min_x = df['sat_min_x'].iloc[0]
                sat_max_x = df['sat_max_x'].iloc[0]
                sat_min_y = df['sat_min_y'].iloc[0]
                sat_max_y = df['sat_max_y'].iloc[0]

                sat_bounds_tweaked = [sat_min_x, sat_min_y, sat_max_x, sat_max_y]

                # load the image
                image = liff.load_image_from_file(image_id, catch=catch,
                                                  verbose=False, pbar=pbar)

                if image is None:
                    p.print_v(f"Error loading {image_id}", verbose=verbose, pbar=None, color="red")
                    continue

                # get the border dimensions
                _, edge_dims = rb.remove_borders(image, image_id=image_id,
                                                 return_edge_dims=True,
                                                 catch=catch, verbose=False, pbar=pbar)

                # remove the border from the small image and then rotate
                image = image[edge_dims[2]:edge_dims[3], edge_dims[0]:edge_dims[1]]
                image = ri.rotate_image(image, angle, start_angle=0)

                if image is None:
                    p.print_v(f"Error rotating {image_id}", verbose=verbose, pbar=None, color="red")
                    continue

            #  we need to find the tps
            else:

                # if true we only want to use images with tps already existing
                if debug_only_existing_tps:
                    p.print_v(f"{image_id} is skipped because it doesn't have tps already calculated",
                              verbose=verbose, pbar=pbar, color="red")
                    continue

                # first we need the approx approx_footprint (if not available, we need to calculate it)
                if data['footprint_type'] == "image":
                    footprint_approx = shapely.wkt.loads(data['footprint'])
                # if there's no approx approx_footprint or we want to recalculate it
                elif overwrite is True or data['footprint_approx'] is None:

                    # for the calculation of the approx approx_footprint we need some more information
                    focal_length = data["focal_length"]
                    height = data["height"]
                    x = data["x_coords"]
                    y = data["y_coords"]
                    view_direction = data["view_direction"]

                    # we need a focal length
                    if focal_length is None:
                        # we don't want to use avg values -> this image we cannot use
                        if use_avg_values is False:
                            p.print_v(f"{image_id} is missing a focal length", verbose=verbose,
                                      pbar=pbar, color="red")
                            continue

                        # we use an average value
                        else:
                            focal_length = 152.457

                    # we need a height
                    if height is None or np.isnan(height):
                        # we don't want to use avg values -> this image we cannot use
                        if use_avg_values is False:
                            p.print_v(f"{image_id} is missing a height", verbose=verbose,
                                      pbar=pbar, color="red")
                            continue

                        # we use an average value
                        else:
                            height = 25000

                    # we need an angle (and that we cannot substitute)
                    if angle is None:
                        p.print_v(f"{image_id} is missing an angle", verbose=verbose,
                                  pbar=pbar, color="red")
                        continue

                    # calculate an approximate approx_footprint based on the information we have from the shapefile
                    footprint_approx = cfa.calc_footprint_approx(x, y, angle, view_direction,
                                                                 height, focal_length,
                                                                 catch=True, verbose=False, pbar=pbar)

                    # sometimes we cannot calculate the approx approx_footprint
                    if footprint_approx is None:
                        p.print_v(f"{image_id} has no approx footprint", verbose=verbose,
                                  pbar=pbar, color="red")
                        continue

                    # save the approx approx_footprint in a shapefile if we want
                    if debug_save_approx_footprints_shape:
                        shape_path = path_georef_overview + "/approx/approx_footprints.shp"

                        # Create a GeoDataFrame with the approx_footprint
                        footprint_gdf = gpd.GeoDataFrame(geometry=[footprint_approx])
                        footprint_gdf.crs = "EPSG:3031"
                        footprint_gdf['image_id'] = image_id
                        footprint_gdf['complexity'] = data['complexity']

                        # Create the shapefile if it doesn't exist
                        if not os.path.exists(shape_path):

                            footprint_gdf.to_file(shape_path)

                        else:
                            # Read the existing shapefile
                            existing_gdf = gpd.read_file(shape_path)

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
                            existing_gdf.to_file(shape_path)

                    # save the approx photocenter in a shapefile if we want
                    if debug_save_approx_photocenter_shape:
                        shape_path = path_georef_overview + "/approx/approx_photocenters.shp"

                        # to save the photocenter we need to calculate it from the approx_footprint
                        x, y = ccp.calc_camera_position(footprint_approx)

                        # Create a GeoDataFrame with the photocenter
                        photocenter_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x], [y]))
                        photocenter_gdf.crs = "EPSG:3031"
                        photocenter_gdf['image_id'] = image_id

                        # Create the shapefile if it doesn't exist
                        if not os.path.exists(shape_path):
                            photocenter_gdf.to_file(shape_path)
                        else:
                            # Read the existing shapefile
                            existing_gdf = gpd.read_file(shape_path)

                            # Check if the image_id is already in our existing photocenters
                            if image_id in existing_gdf['image_id'].values:
                                # Update the entry in existing_gdf
                                existing_gdf.loc[existing_gdf['image_id'] == image_id,
                                                 "geometry"] = photocenter_gdf.iloc[0]['geometry']
                            else:
                                # Append the new photocenter to the existing GeoDataFrame
                                existing_gdf = pd.concat([existing_gdf, photocenter_gdf], ignore_index=True)

                            # Save the updated GeoDataFrame to the shapefile
                            existing_gdf.to_file(shape_path)
                # use existing approx approx_footprint
                else:
                    footprint_approx = shapely.wkt.loads(data['footprint_approx'])

                # check if the image is complex enough to georeference
                if data["complexity"] <= min_complexity:
                    p.print_v(f"{image_id} is too simple for geo-referencing ({data['complexity']})",
                              verbose=verbose, pbar=None, color="red")
                    continue

                # buffer the approx_footprint
                footprint_buffered = bf.buffer_footprint(footprint_approx, buffer_val=buffer_val,
                                                         catch=False, verbose=False, pbar=pbar)

                # if we couldn't buffer the approx_footprint something went wrong
                if footprint_buffered is None:
                    p.print_v(f"{image_id} does not have a buffered footprint", verbose=verbose,
                              pbar=pbar, color="red")
                    continue

                # get the satellite bounds from this approx_footprint
                sat_bounds_buffered = list(footprint_buffered.bounds)
                sat_bounds_buffered = [int(x) for x in sat_bounds_buffered]

                # use information from already geo-referenced image from this flight-path
                # but not if we already use information from images
                if debug_use_existing_information and data["footprint_type"] != "image":

                    # already init the error vectors
                    x_vec, y_vec = 0, 0

                    # get the flight of this image
                    flight = image_id[2:9]
                    image_nr = int(image_id[-4:])

                    # get the data from database
                    sql_string = "SELECT image_id, position_error_vector FROM images_extracted " \
                                 f"WHERE SUBSTRING(image_id, 3, 7) = '{flight}'"
                    ref_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

                    # drop rows without exact approx_footprint
                    ref_data = ref_data.dropna()

                    # we only can calculate an error vector if we have data
                    if ref_data.shape[0] > 0:

                        # get x and y as separate columns
                        ref_data[['x', 'y']] = ref_data['position_error_vector'].str.split(';', expand=True)
                        ref_data['x'] = ref_data['x'].astype(float)
                        ref_data['y'] = ref_data['y'].astype(float)

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
                            subset_ref_data = ref_data[ref_data['image_id'].isin(subset_ids)]

                            # if the order is too small it can happen that we don't have any images
                            if subset_ref_data.shape[0] == 0:
                                x_vec = 0
                                y_vec = 0
                            else:
                                x_vec = round(subset_ref_data['x'].mean(), 4)
                                y_vec = round(subset_ref_data['y'].mean(), 4)

                        # calculate using avg (or both, but only if we didn't already found an error vector)
                        if error_vector_calculation in ["avg", "both"] and x_vec == 0 and y_vec == 0:
                            x_vec = round(ref_data['x'].mean(), 4)
                            y_vec = round(ref_data['y'].mean(), 4)

                    if x_vec != 0 or y_vec != 0:
                        p.print_v("Existing information is used for pre-localization!", verbose=verbose, pbar=pbar)
                    else:
                        p.print_v("No existing information is available for pre-localization", verbose=verbose,
                                  pbar=pbar)

                    sat_bounds_buffered[0] = sat_bounds_buffered[0] + x_vec
                    sat_bounds_buffered[1] = sat_bounds_buffered[1] + y_vec
                    sat_bounds_buffered[2] = sat_bounds_buffered[2] + x_vec
                    sat_bounds_buffered[3] = sat_bounds_buffered[3] + y_vec

                # load the initial approx satellite image
                sat_buffered, sat_transform_buffered = lsd.load_satellite_data(sat_bounds_buffered,
                                                                               month=data["date_month"],
                                                                               fallback_month=True,
                                                                               return_transform=True,
                                                                               catch=catch, verbose=False,
                                                                               pbar=pbar)

                # if the satellite is none something went wrong
                if sat_buffered is None:
                    p.print_v(f"{image_id} does not have a satellite image", verbose=verbose,
                              pbar=pbar, color="red")
                    continue

                # load the image
                image_with_borders = liff.load_image_from_file(image_id, catch=catch, verbose=False, pbar=pbar)

                # if image is none something went wrong
                if image_with_borders is None:
                    p.print_v(f"{image_id} does not have an image", verbose=verbose,
                              pbar=pbar, color="red")
                    continue

                # adjust the image so that pixel size is similar to satellite image
                image_adjusted = air.adjust_image_resolution(sat_buffered,
                                                             sat_bounds_buffered,
                                                             image_with_borders,
                                                             footprint_approx.bounds)

                # get the border dimensions
                img_no_borders, edge_dims = rb.remove_borders(image_with_borders, image_id=image_id,
                                                              return_edge_dims=True,
                                                              catch=catch, verbose=False, pbar=pbar)

                # adapt the border dimension to account for smaller image
                change_y = image_with_borders.shape[0] / image_adjusted.shape[0]
                change_x = image_with_borders.shape[1] / image_adjusted.shape[1]
                if change_y > 0 or change_x > 0:
                    change_y = 1 / change_y
                    change_x = 1 / change_x
                edge_dims[0] = int(edge_dims[0] * change_x)
                edge_dims[1] = int(edge_dims[1] * change_x)
                edge_dims[2] = int(edge_dims[2] * change_y)
                edge_dims[3] = int(edge_dims[3] * change_y)

                # copy to be on the save side
                image = copy.deepcopy(image_adjusted)

                # remove the border from the small image and then rotate
                image = image[edge_dims[2]:edge_dims[3], edge_dims[0]:edge_dims[1]]
                image = ri.rotate_image(image, angle, start_angle=0)

                if debug_show_images:
                    di.display_images([sat_buffered, image], title=f"{image_id}")

                # try to locate the image in a bigger area than just the buffered approx_footprint
                if debug_locate_image:

                    # get the width and height of the satellite image
                    sat_width = sat_buffered.shape[1] * np.abs(sat_transform_buffered[0])
                    sat_height = sat_buffered.shape[2] * np.abs(sat_transform_buffered[4])

                    # calculate the step size
                    step_x = int((1 - overlap) * sat_width)
                    step_y = int((1 - overlap) * sat_height)

                    # save which combinations did we already check
                    lst_checked_combinations = []

                    # iterate the images around our image (starting with center)
                    for order in range(max_order):

                        # create a combinations dict
                        combinations = []
                        for i in range(-order * step_x, (order + 1) * step_x, step_x):
                            for j in range(-order * step_y, (order + 1) * step_y, step_y):
                                combinations.append([i, j])

                        # save our best values
                        best_combination = None
                        best_nr_of_points = 0
                        best_avg_conf = 0

                        best_sat_image = None
                        best_sat_transform = None
                        best_sat_bounds = None

                        # check all combinations
                        for combination in combinations:

                            # check if we already checked this combination
                            if combination in lst_checked_combinations:
                                continue
                            else:
                                lst_checked_combinations.append(combination)

                            p.print_v(f"Check combination {combination} for {image_id}, (Order {order})",
                                      verbose=verbose, pbar=pbar)

                            # adapt the satellite bounds for this combination
                            sat_bounds_adap = copy.deepcopy(sat_bounds_buffered)

                            # first adapt x
                            sat_bounds_adap[0] = sat_bounds_adap[0] + combination[0]
                            sat_bounds_adap[2] = sat_bounds_adap[2] + combination[0]

                            # and then y
                            sat_bounds_adap[1] = sat_bounds_adap[1] + combination[1]
                            sat_bounds_adap[3] = sat_bounds_adap[3] + combination[1]

                            # ignore mean of empty slice & invalid divide value warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")

                                # get the satellite image
                                sat_image_adap, sat_transform_adap = lsd.load_satellite_data(sat_bounds_adap,
                                                                                             return_transform=True,
                                                                                             catch=catch,
                                                                                             verbose=False,
                                                                                             pbar=pbar)

                                # get tie-points between satellite image and historical image
                                tps, conf = ftp.find_tie_points(sat_image_adap, image,
                                                                catch=True, verbose=False, pbar=None)

                                # get number of tie points
                                if tps is None:
                                    nr_adapted_points = 0
                                else:
                                    nr_adapted_points = tps.shape[0]

                                # get average conf
                                if conf is None:
                                    adapted_avg_conf = 0
                                else:
                                    adapted_avg_conf = np.mean(np.asarray(conf))
                                    if np.isnan(adapted_avg_conf):
                                        adapted_avg_conf = 0

                                # check if we have the best combination
                                if success_criteria == "num_points":
                                    if nr_adapted_points > best_nr_of_points or (nr_adapted_points == best_nr_of_points
                                                                                 and best_avg_conf > adapted_avg_conf):
                                        best_combination = combination
                                        best_nr_of_points = nr_adapted_points
                                        best_avg_conf = adapted_avg_conf

                                        best_sat_image = copy.deepcopy(sat_image_adap)
                                        best_sat_transform = copy.deepcopy(sat_transform_adap)
                                        best_sat_bounds = copy.deepcopy(sat_bounds_adap)

                                elif success_criteria == "confidence":
                                    if adapted_avg_conf > best_avg_conf or (adapted_avg_conf == best_avg_conf and
                                                                            nr_adapted_points > best_nr_of_points):
                                        best_combination = combination
                                        best_nr_of_points = nr_adapted_points
                                        best_avg_conf = adapted_avg_conf

                                        best_sat_image = copy.deepcopy(sat_image_adap)
                                        best_sat_transform = copy.deepcopy(sat_transform_adap)
                                        best_sat_bounds = copy.deepcopy(sat_bounds_adap)

                            p.print_v(f"{tps.shape[0]} tie-points found", verbose=verbose, pbar=pbar)

                        # do we already have a nice overlap ->
                        if success_criteria == "num_points" and best_nr_of_points > 25:
                            break
                        elif success_criteria == "confidence" and best_avg_conf > 0.9:
                            break

                    p.print_v(f"Best combination is {best_combination} with "  # noqa
                              f"{best_nr_of_points} tie-points", verbose=verbose, pbar=pbar)  # noqa

                    # get the best satellite image
                    sat_located = copy.deepcopy(best_sat_image)  # noqa
                    sat_transform_located = copy.deepcopy(best_sat_transform)  # noqa
                    sat_bounds_located = copy.deepcopy(best_sat_bounds)  # noqa

                    # with too few tie-points we cannot georeference -> skip this image
                    if best_nr_of_points == 0:
                        p.print_v(f"{image_id} has no points for geo-referencing!", verbose=verbose, pbar=None,
                                  color="red")
                        continue

                # if we don't want to localize
                else:

                    # just copy the original satellite image
                    sat_located = copy.deepcopy(sat_buffered)
                    sat_transform_located = copy.deepcopy(sat_transform_buffered)
                    sat_bounds_located = copy.deepcopy(sat_bounds_buffered)


                #should we tweak the images
                if debug_tweak_image:

                    p.print_v("Start tweaking images", verbose=verbose, pbar=pbar)

                    # initial parameters for tweaking
                    tweaked_best_nr_points = 0

                    # get the values from located
                    sat_image_tweaked = copy.deepcopy(sat_located)
                    sat_transform_tweaked = copy.deepcopy(sat_transform_located)
                    sat_bounds_tweaked = copy.deepcopy(sat_bounds_located)

                    # how often did we tweaked the image
                    counter = 0

                    # tweak the image
                    while True:

                        # we only want to tweak the image a maximum number of times
                        counter = counter + 1
                        if counter == max_tweaks:
                            break

                        p.print_v(f"Tweak image ({counter}/{max_tweaks})", verbose=verbose, pbar=pbar)

                        # get tie-points between satellite image and historical image
                        tps, conf = ftp.find_tie_points(sat_image_tweaked, image,
                                                        catch=True, verbose=False, pbar=None)

                        p.print_v(f"  {tps.shape[0]} points found", verbose=verbose, pbar=pbar)

                        # if the number of points is going up we need to save the params
                        if tps.shape[0] >= tweaked_best_nr_points:
                            # save the best params
                            tweaked_best_nr_points = tps.shape[0]
                            sat_image_tweaked_best = copy.deepcopy(sat_image_tweaked)
                            sat_transform_tweaked_best = copy.deepcopy(sat_transform_tweaked)
                            sat_bounds_tweaked_best = copy.deepcopy(sat_bounds_tweaked)

                        # points are going down
                        else:
                            p.print_v(f"Points going down ({tps.shape[0]} < {tweaked_best_nr_points})",
                                      verbose=verbose, pbar=pbar)

                            # restore the best settings again
                            sat_image_tweaked = copy.deepcopy(sat_image_tweaked_best)
                            sat_transform_tweaked = copy.deepcopy(sat_transform_tweaked_best)
                            sat_bounds_tweaked = copy.deepcopy(sat_bounds_tweaked_best)  # noqa

                            # stop the loop
                            break

                        # if we didn't find any points we can leave
                        if tps.shape[0] == 0:
                            p.print_v(f"No points found", verbose=verbose, pbar=pbar)
                            break

                        # sad, we have too few points -> stop the tweaking
                        if tps.shape[0] < min_nr_of_points:
                            p.print_v(f"Too few tie-points ({tps.shape[0]}) found "
                                      f"for {image_id}", verbose=verbose, pbar=pbar)
                            break

                        # find the direction in which we need to change the satellite image
                        step_y, step_x = ffd.find_footprint_direction(image, tps[:, 2:4], conf)

                        # tweak the satellite bounds
                        sat_bounds_tweaked[0] = sat_bounds_tweaked[0] + step_x
                        sat_bounds_tweaked[1] = sat_bounds_tweaked[1] + step_y
                        sat_bounds_tweaked[2] = sat_bounds_tweaked[2] + step_x
                        sat_bounds_tweaked[3] = sat_bounds_tweaked[3] + step_y

                        # get the tweaked satellite image
                        sat_image_tweaked, sat_transform_tweaked = lsd.load_satellite_data(sat_bounds_tweaked,
                                                                                           month=month,
                                                                                           fallback_month=True,
                                                                                           return_transform=True,
                                                                                           catch=True, verbose=verbose,
                                                                                           pbar=pbar)

                        # catch if something went wrong loading the satellite image
                        if sat_image_tweaked is None:
                            p.print_v("Failed: load tweaked satellite image", verbose=verbose, pbar=pbar)
                            break

                # of we don't want to tweak
                else:
                    sat_image_tweaked = copy.deepcopy(sat_located)
                    sat_transform_tweaked = copy.deepcopy(sat_transform_located)
                    sat_bounds_tweaked = copy.deepcopy(sat_bounds_located)  # noqa

                # a final finding of the tie-points (required if we do not locate and tweak)
                tps, conf = ftp.find_tie_points(sat_image_tweaked, image,
                                                catch=True, verbose=False, pbar=None)

                # with too few tie-points we cannot georeference -> skip this image
                if tps.shape[0] < min_nr_of_points:
                    p.print_v(f"{image_id} has too few points for geo-referencing "
                              f"(before outlier-filtering)!", verbose=verbose, pbar=None, color="red")
                    continue

                p.print_v(f"{tps.shape[0]} points were found", verbose=verbose, pbar=pbar)

                # if wished we can look for outliers in the data
                if debug_filter_outliers:
                    _, mask = cv2.findHomography(tps[:, 0:2], tps[:, 2:4], cv2.RANSAC, 5.0)
                    tps = tps[mask.flatten() == 0]
                    conf = conf[mask.flatten() == 0]
                    p.print_v(f"{np.count_nonzero(mask.flatten())} entries are removed as outliers",
                              verbose=verbose, pbar=pbar)

                    p.print_v(f"{tps.shape[0]} points were found (after outlier filtering)", verbose=verbose, pbar=pbar)

                if tps.shape[0] < min_nr_of_points:
                    p.print_v(f"{image_id} has too few points for geo-referencing!", verbose=verbose, pbar=None, color="red")
                    continue

                # currently the images are adjusted for satellite resolution -> they're smaller
                # if this is false, we enlarge them again
                if debug_use_small_image_size is False:
                    # get the image with original image size
                    image = copy.deepcopy(img_no_borders)

                    # we still need to rotate it
                    image = ri.rotate_image(image, angle, start_angle=0)

                    # adapt the tps to account for the original image-size
                    tps[:, 2] = (tps[:, 2] * (1 / change_x)).astype(int)
                    tps[:, 3] = (tps[:, 3] * (1 / change_y)).astype(int)

                # save images with the tie-points
                if debug_save_tps_images:
                    dt.display_tiepoints([sat_image_tweaked, image], points=tps, confidences=conf,
                                         title=f"Tie points ({image_id})", save_path=path_georef_tie_points_img + f"/sat/{image_id}.png")

                # copy the tie-points
                tps_abs = copy.deepcopy(tps)

                # project satellite points to absolute coords
                tps_abs[:, 0] = tps_abs[:, 0] * sat_transform_tweaked[0]
                tps_abs[:, 1] = tps_abs[:, 1] * sat_transform_tweaked[4]
                tps_abs[:, 0] = tps_abs[:, 0] + sat_transform_tweaked[2]
                tps_abs[:, 1] = sat_transform_tweaked[5] - sat_image_tweaked.shape[0] * sat_transform_tweaked[
                    4] + tps_abs[:, 1]

                # save the points to a shape file
                if debug_save_tps_shape:

                    # Create a DataFrame from the NumPy array
                    data = {'x': tps_abs[:, 0], 'y': tps_abs[:, 1],
                            'x_sat': tps[:, 0], 'y_sat': tps[:, 1],
                            'x_pic': tps[:, 2], 'y_pic': tps[:, 3],
                            'conf': conf,
                            "pic_x_fac": change_x, "pic_y_fac": change_y,
                            'sat_min_x': sat_bounds_tweaked[0], 'sat_max_x': sat_bounds_tweaked[2],
                            'sat_min_y': sat_bounds_tweaked[1], 'sat_max_y': sat_bounds_tweaked[3]}
                    df = pd.DataFrame(data)

                    geometry = gpd.points_from_xy(x=df['x'], y=df['y'])
                    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:3031')

                    gdf.to_file(path_georef_tie_points_shp + f"/sat/{image_id}_points.shp", driver='ESRI Shapefile')

            # do we want to see the tie-points
            if debug_show_tps:
                sat_image = lsd.load_satellite_data(sat_bounds_tweaked, verbose=verbose)

                dt.display_tiepoints([sat_image, image], points=tps, confidences=conf,
                                     title=f"Tie points ({image_id})")

            # do we want to geo-reference
            if debug_georeference:

                # get the transform
                transform = ag.apply_gpcs(image_id, image, tps_abs, path_georef_tiffs,
                                                  transform_method, gdal_order)

                if transform is None:
                    p.print_v(f"Something went wrong applying gcps to {image_id}", color="red",
                              verbose=verbose, pbar=pbar)
                    continue

                # should we check the images after geo-referencing?
                if debug_final_check:

                    # check if the image would work
                    good_image = cgi.check_georef_image(path_georef_tiffs + "/" + image_id + ".tif")

                    # remove potential leftovers from the folders and shapefiles
                    if good_image is False:
                        try:
                            # move tiff to failed
                            os.rename(path_georef_tiffs + f"/sat/{image_id}.tif", path_georef_tiffs + f"/sat_failed/{image_id}.tif")

                            # move tie-points to failed
                            os.rename(path_georef_tie_points_img + f"/sat/{image_id}.png", path_georef_tie_points_img + f"/sat_failed/{image_id}.png")
                            os.rename(path_georef_tie_points_shp + f"/sat{image_id}_points.shp", path_georef_tie_points_shp + f"/sat_failed/{image_id}_points.shp")

                            # remove from table
                            if data['footprint_type'] != "image":
                                sql_string = f"UPDATE images_extracted SET " \
                                             f"footprint = NULL, position_exact = NULL, " \
                                             f"position_error_vector = NULL, footprint_type = NULL " \
                                             f"WHERE image_id='{image_id}'"
                                ctd.edit_data_in_db(sql_string, catch, verbose, pbar)

                            # remove from the overview
                            error_vectors_gpd = gpd.read_file(path_georef_overview + "/sat/sat_error_vectors.shp")
                            footprints_gpd = gpd.read_file(path_georef_overview + "/sat/sat_footprints.shp")
                            photocenters_gpd = gpd.read_file(path_georef_overview + "/sat/sat_photocenters.shp")

                            ev_cond = (error_vectors_gpd['image_id'] == image_id)
                            f_cond = (footprints_gpd['image_id'] == image_id)
                            p_cond = (photocenters_gpd['image_id'] == image_id)

                            error_vectors_gpd = error_vectors_gpd[~ev_cond]
                            footprints_gpd = footprints_gpd[~f_cond]
                            photocenters_gpd = photocenters_gpd[~p_cond]

                            error_vectors_gpd.to_file(path_georef_overview + "/sat/sat_error_vectors.shp")
                            footprints_gpd.to_file(path_georef_overview + "/sat/sat_footprints.shp")
                            photocenters_gpd.to_file(path_georef_overview + "/sat/sat_photocenters.shp")

                        except:
                            pass

                # if we don't want to check the images, they're always right
                else:
                    good_image = True

                # should we save the images in the database
                if debug_save_exact_in_database and good_image:
                    sf.save_footprint(image_id, footprint_type="satellite", overwrite=overwrite, verbose=verbose, pbar=pbar)

                if good_image:
                    p.print_v(f"{image_id} successfully georeferenced!", color="green", verbose=verbose, pbar=pbar)
                    nr_new_images += 1
                else:
                    p.print_v(f"{image_id} falsely georeferenced!", color="red", verbose=verbose, pbar=pbar)

                p.print_v(f"{nr_new_images} already additionally matched!", verbose=verbose, pbar=pbar)

        except (Exception,) as e:
            if catch:
                p.print_v(f"Image {image_id} failed", verbose=verbose, pbar=pbar, color="red")
                continue
            else:
                raise e

    print(nr_new_images, "new georeferenced!!")


if __name__ == "__main__":
    _sql_string = "SELECT image_id FROM images WHERE view_direction = 'V'"
    ids = ctd.get_data_from_db(_sql_string).values.tolist()
    ids = [item for sublist in ids for item in sublist]

    # ids = []
    # folder_path = "/data_1/ATM/data_1/playground/georef2/tiffs/georeferenced"
    # for file_name in os.listdir(folder_path):
    #
    #    ids.append(file_name[:-4])

    # ids = ["CA216632V0306"]
    # ids = ["CA183332V0097"]
    # ids = ["CA512732V0126"]
    # ids = ["CA184532V0231"]
    # ids = ["CA182732V0020"]
    #ids = ["CA181332V0128"]

    import random

    random.shuffle(ids)

    georef2(ids, overwrite=_overwrite, catch=_catch, verbose=_verbose)
