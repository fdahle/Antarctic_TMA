import cv2 as cv2
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio

from rasterio.io import MemoryFile
from shapely.geometry import Point

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.rotate_image as ri

import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.calc_camera_position as ccp
import image_georeferencing.sub.check_georef_image as cgi
import image_georeferencing.sub.convert_image_to_footprint as citf

import image_tie_points.find_tie_points as ftp

import display.display_images as di
import display.display_tiepoints as dt

from tqdm import tqdm

# general settings
_overwrite = False
_catch = False
_verbose = True

# function settings
filter_outliers = True
min_threshold = 0.8  # the minimum confidence of tie-points
min_nr_of_tps = 25  # how many tie-points we need minimum for georef
transform_method = "rasterio"  # should 'rasterio' or 'gdal' be used for geo-referencing
transform_order = 2  # which order of transform should be used

# function params
debug_check_georef = True  # should geo-referenced images be checked for validity
debug_create_masks = True
debug_georeference = True

# show params
debug_show_tie_points = True
debug_show_masks = False

# save params
debug_save_in_db = True
debug_save_footprints_shape = True
debug_save_photocenters_shape = True
debug_save_error_vector_shape = True
debug_save_tie_points_image = True
debug_save_tie_points_shape = True

# print params
debug_print_all = True  # if true, tqdm is not used

# here we want to store the created data
base_path = "/data_1/ATM/data_1/playground/georef2"


def georef_img(image_ids, overwrite=False, catch=True, verbose=False):
    print(f"georeference {len(image_ids)} images")

    # this variable will contain the number of images we successfully georeferenced
    nr_new_images = 0

    # path of geo-referenced images from satellite
    ref_path = "/data_1/ATM/data_1/playground/georef2/tiffs/sat"

    # overview paths
    path_exact_footprints_shape = base_path + "/overview/img/img_footprints.shp"
    path_exact_footprints_shape_failed = base_path + "/overview/img_failed/img_footprints_failed.shp"
    path_exact_photocenters_shape = base_path + "/overview/img/img_photocenters.shp"
    path_exact_photocenters_shape_failed = base_path + "/overview/img_failed/img_photocenters_failed.shp"

    # iterate all images
    for image_id in (pbar := tqdm(image_ids)):

        if debug_print_all:
            pbar = None

        p.print_v(f"georeference {image_id}", verbose=verbose, pbar=pbar)

        # tiff paths
        path_tiff = base_path + f"/tiffs/img/{image_id}.tif"
        path_tiff_failed = base_path + f"/tiffs/img_failed/{image_id}.tif"

        # tie point paths
        path_tie_points_img = base_path + f"/tie_points/images/img/{image_id}.png"
        path_tie_points_img_failed = base_path + f"/tie_points/images/img_failed/{image_id}.png"
        path_tie_points_shp = base_path + f"/tie_points/shapes/img/{image_id}_points.shp"
        path_tie_points_shp_failed = base_path + f"/tie_points/shapes/img/{image_id}_points.shp"

        # get information from table images
        sql_string = f"SELECT * FROM images WHERE image_id='{image_id}'"
        data_images = ctd.get_data_from_db(sql_string, catch=False, verbose=verbose)

        # get information from table images_extracted
        sql_string = "SELECT image_id, height, focal_length, complexity, text_bbox, " \
                     "ST_AsText(footprint_approx) AS footprint_approx, " \
                     "ST_AsText(footprint) AS footprint, " \
                     "footprint_type AS footprint_type " \
                     f"FROM images_extracted WHERE image_id ='{image_id}'"
        data_extracted = ctd.get_data_from_db(sql_string, catch=False, verbose=verbose)

        # merge the data from both tables
        data = pd.merge(data_images, data_extracted, on='image_id').iloc[0]

        # First, we check if the image is already georeferenced
        if overwrite is False and data['footprint'] is not None and \
                data['footprint_type'] == "satellite":
            p.print_v(f"{image_id} is already georeferenced!", verbose=verbose,
                      pbar=None, color="green")
            continue

        # check if the image is a failed image
        if os.path.isfile(path_tiff_failed):

        # get the direct neighbours
        image_nr = int(image_id[-4:])
        prev_id = image_id[:-4] + '{:04d}'.format(int(image_nr) - 1)
        next_id = image_id[:-4] + '{:04d}'.format(int(image_nr) + 1)

        # get the data of the neighbours
        prev_sql_string = f"SELECT * FROM images_extracted WHERE image_id='{prev_id}'"
        next_sql_string = f"SELECT * FROM images_extracted WHERE image_id='{next_id}'"
        prev_data = ctd.get_data_from_db(prev_sql_string, catch=catch, verbose=verbose, pbar=pbar)
        next_data = ctd.get_data_from_db(next_sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # check if we have a approx_footprint for a neighbour
        if prev_data.shape[0] > 0:
            prev_data = prev_data.iloc[0]
            prev_footprint_exact = prev_data['footprint']
        else:
            prev_footprint_exact = None
        if next_data.shape[0] > 0:
            next_data = next_data.iloc[0]
            next_footprint_exact = next_data['footprint']
        else:
            next_footprint_exact = None

        # if both neighbours are not georeferenced -> skip this image
        if prev_footprint_exact is None and next_footprint_exact is None:
            p.print_v("No direct neighbours are georeferenced.", verbose=verbose, pbar=pbar)
            continue

        # load base image and base mask
        image = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)

        # load prev image
        if prev_footprint_exact is not None:
            prev_path = ref_path + "/" + prev_id + ".tif"

            # safety check if the image is really existing
            if os.path.isfile(prev_path) is False:
                prev_footprint_exact = None
            else:
                prev_image, prev_transform = liff.load_image_from_file(prev_path, return_transform=True,
                                                                       catch=catch, verbose=verbose, pbar=pbar)
        # load next image
        if next_footprint_exact is not None:
            next_path = ref_path + "/" + next_id + ".tif"

            # safety check if the image is really existing
            if os.path.isfile(next_path) is False:
                next_footprint_exact = None
            else:
                next_image, next_transform = liff.load_image_from_file(next_path, return_transform=True,
                                                                       catch=catch, verbose=verbose, pbar=pbar)

        # should we mask the images when looking for tps?
        if debug_create_masks:

            # get the borders of the image
            image_no_borders, border_dims = rb.remove_borders(image, image_id, return_edge_dims=True)

            # create mask and remove borders
            mask = np.ones_like(image)
            mask[:border_dims[2], :] = 0
            mask[border_dims[3]:, :] = 0
            mask[:, :border_dims[0]] = 0
            mask[:, border_dims[1]:] = 0

            # get the text boxes
            text_boxes = data['text_bbox'].split(";")

            # Convert string representation to list of tuples
            coordinate_pairs = [tuple(map(int, coord[1:-1].split(','))) for coord in text_boxes]

            # Set the regions in the array to 0
            for top_left_x, top_left_y, bottom_right_x, bottom_right_y in coordinate_pairs:
                mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

            # the image and the mask must be rotated
            image = ri.rotate_image(image, data['azimuth'], catch=catch, verbose=False, pbar=pbar)
            mask = ri.rotate_image(mask, data['azimuth'], catch=catch, verbose=False, pbar=pbar)

            # create mask for prev image
            if prev_footprint_exact is not None:

                # we need the unreferenced image as well for mask creation
                prev_image_unreferenced = liff.load_image_from_file(prev_id, catch=catch,
                                                                    verbose=verbose, pbar=pbar)

                # create mask
                prev_mask_temp = np.ones_like(prev_image_unreferenced)

                # get the single text boxes
                prev_boxes = prev_data['text_bbox'].split(";")

                # Convert string representation to list of tuples
                prev_pairs = [tuple(map(int, coord[1:-1].split(','))) for coord in prev_boxes]

                # Set the regions in the array to 0
                for top_left_x, top_left_y, bottom_right_x, bottom_right_y in prev_pairs:
                    # Apply the transform to the tuple of coordinates
                    prev_mask_temp[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

                # remove the borders
                prev_mask_temp = rb.remove_borders(prev_mask_temp, prev_id)

                # get the angle
                sql_string = f"SELECT azimuth FROM images WHERE image_id='{prev_id}'"
                prev_m_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                prev_angle = prev_m_data['azimuth'].iloc[0]

                # rotate
                prev_mask_temp = ri.rotate_image(prev_mask_temp, prev_angle, start_angle=0)

                # transform the mask
                # Open a new in-memory file for writing.
                with MemoryFile() as memfile:
                    with memfile.open(driver='GTiff',
                                      height=prev_image.shape[0], width=prev_image.shape[1],  # noqa
                                      count=1, dtype=str(prev_mask_temp.dtype),
                                      crs='EPSG:3031', transform=prev_transform) as dataset:  # noqa
                        # Write the mask to the in-memory file.
                        dataset.write(prev_mask_temp, 1)

                    # Open the in-memory file for reading.
                    with memfile.open() as dataset:
                        # Now, dataset is a rasterio dataset that you can use as you would
                        # use a dataset opened from a file on disk.

                        prev_mask = dataset.read()[0]

            # create mask for next image
            if next_footprint_exact is not None:

                # we need the unreferenced image as well for mask creation
                next_image_unreferenced = liff.load_image_from_file(next_id, catch=catch,
                                                                    verbose=verbose, pbar=pbar)

                # create mask
                next_mask_temp = np.ones_like(next_image_unreferenced)

                # get the single text boxes
                next_boxes = next_data['text_bbox'].split(";")

                # Convert string representation to list of tuples
                next_pairs = [tuple(map(int, coord[1:-1].split(','))) for coord in next_boxes]

                # Set the regions in the array to 0
                for top_left_x, top_left_y, bottom_right_x, bottom_right_y in next_pairs:
                    # Apply the transform to the tuple of coordinates
                    next_mask_temp[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

                # remove the borders
                next_mask_temp = rb.remove_borders(next_mask_temp, next_id)

                # get the angle
                sql_string = f"SELECT azimuth FROM images WHERE image_id='{next_id}'"
                next_m_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                next_angle = next_m_data['azimuth'].iloc[0]

                # rotate
                next_mask_temp = ri.rotate_image(next_mask_temp, next_angle, start_angle=0)

                # transform the mask
                # Open a new in-memory file for writing.
                with MemoryFile() as memfile:
                    with memfile.open(driver='GTiff',
                                      height=next_image.shape[0], width=next_image.shape[1],  # noqa
                                      count=1, dtype=str(next_mask_temp.dtype),
                                      crs='EPSG:3031', transform=next_transform) as dataset:  # noqa
                        # Write the mask to the in-memory file.
                        dataset.write(next_mask_temp, 1)

                    # Open the in-memory file for reading.
                    with memfile.open() as dataset:
                        # Now, dataset is a rasterio dataset that you can use as you would
                        # use a dataset opened from a file on disk.

                        next_mask = dataset.read()[0]

            # show the created masks
            if debug_show_masks:

                # init the lst of images and mask, the image and mask we always have
                lst_images = [image]
                lst_masks = [mask]

                # check if we have a prev approx_footprint, otherwise add fake image and mask
                if prev_footprint_exact is not None:
                    lst_images.append(prev_image)
                    lst_masks.append(prev_mask)
                else:
                    lst_images.append(np.ones([2, 2]))
                    lst_masks.append(np.ones([2, 2]))

                # check if we have a next approx_footprint, otherwise add fake image and mask
                if next_footprint_exact is not None:
                    lst_images.append(next_image)
                    lst_masks.append(next_mask)
                else:
                    lst_images.append(np.ones([2, 2]))
                    lst_masks.append(np.ones([2, 2]))

                di.display_images(lst_images + lst_masks,
                                  title=f"Masks for {image_id}", x_titles=[image_id, "prev", "next"])

        else:

            # we need to rotate the image
            image = ri.rotate_image(image, data['azimuth'], catch=catch, verbose=False, pbar=pbar)

            mask = np.ones_like(image)
            if prev_footprint_exact is not None:
                prev_mask = np.ones_like(prev_image)
            if next_footprint_exact is not None:
                next_mask = np.ones_like(next_image)

        # look for tie-points
        if prev_footprint_exact is not None:

            # find tps between the images
            prev_tps, prev_conf = ftp.find_tie_points(image, prev_image,
                                                      mask_1=mask, mask_2=prev_mask,
                                                      min_threshold=min_threshold,
                                                      catch=catch, verbose=verbose, pbar=pbar)

            if debug_show_tie_points and prev_tps.shape[0] > 0:
                dt.display_tiepoints([image, prev_image], points=prev_tps, confidences=prev_conf,
                                     title=f"Prev tps for {image_id}", verbose=True)

            if debug_save_tie_points_image and prev_tps.shape[0] > 0:
                # adapt the save-path
                prev_save_path = path_tie_points_img[:-3] + "_next.png"

                dt.display_tiepoints([image, prev_image], points=prev_tps, confidences=prev_conf,
                                     title=f"Prev ({prev_id}) tps for {image_id}", verbose=False,
                                     save_path=prev_save_path)

        # look for tie-points
        if next_footprint_exact is not None:

            # find tps between the images
            next_tps, next_conf = ftp.find_tie_points(image, next_image,
                                                      mask_1=mask, mask_2=next_mask,
                                                      min_threshold=min_threshold,
                                                      catch=catch, verbose=verbose, pbar=pbar)

            if debug_show_tie_points and next_tps.shape[0] > 0:
                dt.display_tiepoints([image, next_image], points=next_tps, confidences=next_conf,
                                     title=f"Next ({next_id})tps for {image_id}", verbose=True)

            if debug_save_tie_points_image and next_tps.shape[0] > 0:
                # adapt the save-path
                next_save_path = path_tie_points_img[:-3] + "_next.png"

                dt.display_tiepoints([image, next_image], points=next_tps, confidences=next_conf,
                                     title=f"Next ({next_id}) tps for {image_id}", verbose=False,
                                     save_path=next_save_path)

        # this array will later contain all tie points
        all_tps = np.empty([0, 4])
        all_conf = []

        # check if we can use the prev image for gcps
        if prev_footprint_exact is not None and prev_tps.shape[0] > 0:  # noqa

            # apply transformation to the tie-points of the prev image, so that these tps are absolute
            prev_tps[:, 2:] = np.array(
                rasterio.transform.xy(prev_transform, prev_tps[:, 3], prev_tps[:, 2])).transpose()

            # save these tps
            if debug_save_tie_points_shape:
                # create a pd frame
                data = {'x': prev_tps[:, 2], 'y': prev_tps[:, 3]}
                df = pd.DataFrame(data)

                # create a gpd frame
                geometry = gpd.points_from_xy(x=df['x'], y=df['y'])
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:3031')

                # save gpd frame
                gdf.to_file(f"/data_1/ATM/data_1/playground/georef2/temp/tps/{prev_id}_prev_tps.shp",
                            driver='ESRI Shapefile')

            # save prev tps to all tps
            all_tps = np.concatenate((all_tps, prev_tps), axis=0)

            # save prev conf to all conf
            all_conf = all_conf + prev_conf  # noqa

        # check if we can use the next image for gcps
        if next_footprint_exact is not None and next_tps.shape[0] > 0:  # noqa

            # apply transformation to the tie-points of the next image, so that these tps are absolute
            next_tps[:, 2:] = np.array(
                rasterio.transform.xy(next_transform, next_tps[:, 3], next_tps[:, 2])).transpose()

            # save these tps
            if debug_save_tie_points_shape:
                # create a pd frame
                data = {'x': next_tps[:, 2], 'y': next_tps[:, 3]}
                df = pd.DataFrame(data)

                # create a gpd frame
                geometry = gpd.points_from_xy(x=df['x'], y=df['y'])
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:3031')

                # save gpd frame
                gdf.to_file(f"/data_1/ATM/data_1/playground/georef2/temp/tps/{next_id}_next_tps.shp",
                            driver='ESRI Shapefile')

            # save next tps to all tps
            all_tps = np.concatenate((all_tps, next_tps), axis=0)

            # save next conf to all conf
            all_conf = all_conf + next_conf  # noqa

        # filter for outliers
        if filter_outliers and all_tps.shape[0] >= 4:
            _, mask = cv2.findHomography(all_tps[:, 0:2], all_tps[:, 2:4], cv2.RANSAC, 5.0)
            mask = mask.flatten()

            p.print_v(f"{np.count_nonzero(mask)} outliers removed.", verbose=verbose, pbar=pbar)

            # 1 means outlier
            all_tps = all_tps[mask == 0]
            all_conf = all_conf[mask.flatten() == 0]

        # check if we have enough tie-points
        if all_tps.shape[0] < min_nr_of_tps:
            p.print_v(f"{image_id} has too few ({all_tps.shape[0]}) tie-points",
                      color="red", verbose=verbose, pbar=pbar)
            continue

        # switch tps, so that the geo-referenced tps are on the left side
        tps = np.concatenate((all_tps[:, 2:4], all_tps[:, 0:2]), axis=1)

        # calculate average conf
        avg_conf = np.mean(np.asarray(all_conf))

        if debug_save_tie_points_shape:
            # Create a DataFrame from the NumPy array
            data = {'x': tps[:, 0], 'y': tps[:, 1],
                    'x_pic': tps[:, 2], 'y_pic': tps[:, 3]}
            df = pd.DataFrame(data)

            # create a geopandas frame
            geometry = gpd.points_from_xy(x=df['x'], y=df['y'])
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:3031')

            gdf.to_file(path_tie_points_shp, driver='ESRI Shapefile')

        # the actual geo-referencing
        if debug_georeference:

            # get the transform
            transform = ag.apply_gcps(image_id, image, tps, path_tiff,
                                      transform_method, transform_order)

            if transform is None:
                p.print_v(f"Something went wrong applying gcps to {image_id}", color="red",
                          verbose=verbose, pbar=pbar)
                continue

            # check if the image was geo-referenced correctly
            if debug_check_georef:
                image_is_valid = cgi.check_georef_image(path_tiff, catch=catch, verbose=verbose, pbar=pbar)

            # without checks images are always valid
            else:
                image_is_valid = True

            # for valid images we can create a approx_footprint
            if image_is_valid:

                # increase number of new geo-referenced images
                nr_new_images = nr_new_images + 1

                # get the exact approx_footprint
                footprint_exact = citf.convert_image_to_footprint(image, image_id, transform,
                                                                  catch=catch, verbose=verbose, pbar=pbar)

                # to save the photocenter we need to calculate it from the approx_footprint
                x, y = ccp.calc_camera_position(footprint_exact)

                # create a shapely point that we can save in a db
                photocenter_exact = Point(x, y)

                # save exact approx_footprint in db
                if debug_save_in_db:
                    sql_string = f"UPDATE images_extracted SET " \
                                 f"footprint=ST_GeomFromText('{footprint_exact}'), " \
                                 f"photocenter_exact=ST_GeomFromText('{photocenter_exact}'), " \
                                 f"footprint_type='image' " \
                                 f"WHERE image_id='{image_id}'"
                    success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=False, pbar=None)
                    if success is False:
                        p.print_v(f"Something went wrong saving the exact footprint in db for {image_id}",
                                  verbose=verbose, pbar=pbar, color="red")
                        continue

                # save the exact approx_footprint in a shapefile if we want
                if debug_save_footprints_shape:

                    # Create a GeoDataFrame with the approx_footprint
                    footprint_gdf = gpd.GeoDataFrame(geometry=[footprint_exact])
                    footprint_gdf.crs = "EPSG:3031"
                    footprint_gdf['image_id'] = image_id

                    # add additional information to the approx_footprint
                    footprint_gdf['complexity'] = data['complexity']
                    footprint_gdf['nr_points'] = tps.shape[0]
                    footprint_gdf['conf'] = avg_conf

                    # Create the shapefile if it doesn't exist
                    if not os.path.exists(path_exact_footprints_shape):

                        footprint_gdf.to_file(path_exact_footprints_shape)

                    else:
                        # Read the existing shapefile
                        existing_gdf = gpd.read_file(path_exact_footprints_shape)

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
                        existing_gdf.to_file(path_exact_footprints_shape)

                # save the exact photocenter in a shapefile if we want
                if debug_save_photocenters_shape:

                    # Create a GeoDataFrame with the photocenter
                    photocenter_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([x], [y]))
                    photocenter_gdf.crs = "EPSG:3031"
                    photocenter_gdf['image_id'] = image_id

                    # Create the shapefile if it doesn't exist
                    if not os.path.exists(path_exact_photocenters_shape):
                        photocenter_gdf.to_file(path_exact_photocenters_shape)
                    else:
                        # Read the existing shapefile
                        existing_gdf = gpd.read_file(path_exact_photocenters_shape)

                        # Check if the image_id is already in our existing photocenters
                        if image_id in existing_gdf['image_id'].values:
                            # Update the entry in existing_gdf
                            existing_gdf.loc[existing_gdf['image_id'] == image_id,
                                             "geometry"] = photocenter_gdf.iloc[0]['geometry']
                        else:
                            # Append the new photocenter to the existing GeoDataFrame
                            existing_gdf = pd.concat([existing_gdf, photocenter_gdf], ignore_index=True)

                        # Save the updated GeoDataFrame to the shapefile
                        existing_gdf.to_file(path_exact_photocenters_shape)

            # invalid images we need to relocate and/or delete
            else:

                # relocate images to the failed folder
                try:
                    os.rename(path_tiff, path_tiff_failed)
                except (Exception,):
                    pass

                # relocate tie point images to the failed folder
                try:
                    os.rename(path_tie_points_img, path_tie_points_img_failed)
                except (Exception,):
                    pass

                # relocate tie point shapes to the failed folder
                try:

                    # we need to make the path shorter, as we cannot move only the shp file but others as well
                    short_path = path_tie_points_shp[:-3]
                    short_path_failed = path_tie_points_shp_failed[:-3]

                    # relocate all five files of a shp file
                    os.rename(short_path + "cpg", short_path_failed + "cpg")
                    os.rename(short_path + "dbf", short_path_failed + "dbf")
                    os.rename(short_path + "prj", short_path_failed + "prj")
                    os.rename(short_path + "shp", short_path_failed + "shp")
                    os.rename(short_path + "shx", short_path_failed + "shx")

                except (Exception,):
                    pass

                # remove entry from footprints shape
                try:
                    # open the existing footprints shapefile
                    gdf = gpd.read_file(path_exact_footprints_shape)

                    # get the entry we want to delete
                    entry = gdf[gdf['image_id'] == image_id]

                    if not entry.empty:
                        gdf = gdf[gdf[image_id] != image_id]

                        # Save the existing approx_footprint shapefile
                        gdf.to_file(path_exact_footprints_shape, driver='ESRI Shapefile')

                        # Create the shapefile if it doesn't exist
                        if not os.path.exists(path_exact_footprints_shape_failed):
                            entry.to_file(path_exact_footprints_shape_failed)
                        else:
                            # Read the existing shapefile
                            existing_gdf = gpd.read_file(path_exact_footprints_shape_failed)

                            # Check if the image_id is already in our existing photocenters
                            if image_id in existing_gdf['image_id'].values:
                                # Update the entry in existing_gdf
                                existing_gdf.loc[existing_gdf['image_id'] == image_id,
                                                 "geometry"] = entry
                            else:
                                # Append the new photocenter to the existing GeoDataFrame
                                existing_gdf = pd.concat([existing_gdf, entry], ignore_index=True)

                            # Save the updated GeoDataFrame to the shapefile
                            existing_gdf.to_file(path_exact_footprints_shape_failed, driver='ESRI Shapefile')

                except (Exception,):

                    pass

                # remove entry from photocenter shape
                try:
                    # open the existing photocenters shapefile
                    gdf = gpd.read_file(path_exact_photocenters_shape)

                    # get the entry we want to delete
                    entry = gdf[gdf['image_id'] == image_id]

                    if not entry.empty:
                        gdf = gdf[gdf[image_id] != image_id]

                        # Save the existing photocenter shapefile
                        gdf.to_file(path_exact_photocenters_shape, driver='ESRI Shapefile')

                        # Create the shapefile if it doesn't exist
                        if not os.path.exists(path_exact_photocenters_shape_failed):
                            entry.to_file(path_exact_photocenters_shape_failed)
                        else:
                            # Read the existing shapefile
                            existing_gdf = gpd.read_file(path_exact_photocenters_shape_failed)

                            # Check if the image_id is already in our existing photocenters
                            if image_id in existing_gdf['image_id'].values:
                                # Update the entry in existing_gdf
                                existing_gdf.loc[existing_gdf['image_id'] == image_id,
                                                 "geometry"] = entry
                            else:
                                # Append the new photocenter to the existing GeoDataFrame
                                existing_gdf = pd.concat([existing_gdf, entry], ignore_index=True)

                            # Save the updated GeoDataFrame to the shapefile
                            existing_gdf.to_file(path_exact_photocenters_shape_failed, driver='ESRI Shapefile')

                except (Exception,):

                    pass


if __name__ == "__main__":
    _sql_string = "SELECT image_id FROM images WHERE view_direction = 'V'"
    ids = ctd.get_data_from_db(_sql_string).values.tolist()
    ids = [item for sublist in ids for item in sublist]

    georef_img(ids, overwrite=_overwrite, catch=_catch, verbose=_verbose)
