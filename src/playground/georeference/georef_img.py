import cv2 as cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import os

from rasterio.io import MemoryFile
from tqdm import tqdm

import image_tie_points.find_tie_points as ftp

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.rotate_image as ri

import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.check_georef_image as cgi

import display.display_images as di
import display.display_tiepoints as dt

_catch = False
_verbose = True

debug_print_all = True
debug_create_masks = True  # create substitute masks if we don't have masks
debug_filter_outliers = True  # filter outliers from the tps
debug_remove_border = True  # remove the border from the image we want to georeference
debug_reduce_image_size = True
debug_group_tps = True

debug_save_tps_img = False
debug_save_tps_shape = True
debug_final_check = True
debug_save_georef_tiffs = True

debug_show_tps = False
debug_show_masks = False

start_angle = 0
min_quality = 0.8  # the minimum quality of tps between images (between 0 and 1)
overlap = 0.6  # the estimated overlap between images (used to make images smaller)
min_nr_tps = 5

ref_fld = "/data_1/ATM/data_1/playground/georef2/tiffs/georeferenced/"


def georef_img(image_ids, catch=True, verbose=False):
    print("Start georef3")

    # iterate all images
    for image_id in (pbar := tqdm(image_ids)):

        if debug_print_all:
            pbar = None

        p.print_v(f"Check {image_id}", verbose=verbose, pbar=pbar)

        # get the sql data for this image
        sql_string = f"SELECT * FROM images_extracted WHERE image_id='{image_id}'"
        data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
        data = data.iloc[0]

        # pre-define prev and next image
        prev_image = None
        next_image = None

        # first check if we already have a georeferenced file -> skip this
        if data['footprint'] is not None:
            p.print_v(f"{image_id} is already geo-referenced", verbose=verbose, pbar=pbar)
            continue

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
            prev_footprint_exact = prev_data.iloc[0]['footprint']
        else:
            prev_footprint_exact = None
        if next_data.shape[0] > 0:
            next_footprint_exact = next_data.iloc[0]['footprint']
        else:
            next_footprint_exact = None

        # if both neighbours are not georeferenced -> skip this image
        if prev_footprint_exact is None and next_footprint_exact is None:
            p.print_v("No direct neighbours are georeferenced.", verbose=verbose, pbar=pbar)
            continue

        # get the tie points for the prev image
        if prev_footprint_exact is not None:
            prev_sql_string = f"SELECT * FROM images_tie_points WHERE " \
                              f"(image_1_id='{image_id}' AND image_2_id='{prev_id}') OR " \
                              f"(image_1_id='{prev_id}' AND image_2_id='{image_id}')"
            print("BE CAREFUL !!!!")
            prev_tps = ctd.get_data_from_db(prev_sql_string, catch=False, verbose=verbose, pbar=pbar)
        else:
            prev_tps = pd.DataFrame()

        # get the tie points for the next image
        if next_footprint_exact is not None:
            next_sql_string = f"SELECT * FROM images_tie_points WHERE " \
                              f"(image_1_id='{image_id}' AND image_2_id='{next_id}') OR " \
                              f"(image_1_id='{next_id}' AND image_2_id='{image_id}')"
            next_tps = ctd.get_data_from_db(next_sql_string, catch=catch, verbose=verbose, pbar=pbar)
        else:
            next_tps = pd.DataFrame()

        # init base image and mask (only if we need to get tps for prev or next image)
        if prev_tps.shape[0] == 0 or next_tps.shape[0] == 0:

            # load base image and base mask
            image = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)
            mask = liff.load_image_from_file(image_id, image_path="/data_1/ATM/data_1/aerial/TMA/masked",
                                             catch=True, verbose=verbose, pbar=pbar)

            # if we don't have mask -> create mask
            if mask is None and debug_create_masks:

                # get the borders
                _, edge_dims = rb.remove_borders(image, image_id, return_edge_dims=True)

                # create mask and remove borders
                mask = np.ones_like(image)
                mask[:edge_dims[2], :] = 0
                mask[edge_dims[3]:, :] = 0
                mask[:, :edge_dims[0]] = 0
                mask[:, edge_dims[1]:] = 0

                # get the text boxes
                sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id='{image_id}'"
                m_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                text_boxes = m_data['text_bbox'].iloc[0].split(";")

                # Convert string representation to list of tuples
                coordinate_pairs = [tuple(map(int, coord[1:-1].split(','))) for coord in text_boxes]

                # Set the regions in the array to 0
                for top_left_x, top_left_y, bottom_right_x, bottom_right_y in coordinate_pairs:
                    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
            else:
                p.print_v(f"Mask is None for {image_id}", verbose=verbose, pbar=pbar)
                continue

            # remove the border from the image we want to georeference
            if debug_remove_border:

                # remove the border from the image
                image = rb.remove_borders(image, image_id, catch=catch, verbose=False, pbar=pbar)

                # remove the border from the image
                mask = rb.remove_borders(mask, image_id, catch=catch, verbose=False, pbar=pbar)

            # get the azimuth data of the image
            sql_string = f"SELECT azimuth FROM images WHERE image_id='{image_id}'"
            angle_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=False, pbar=pbar)

            # skip this image if we don't have azimuth data
            if angle_data is None:
                p.print_v(f"There was a problem extracting an angle for {prev_id}",
                          color="red", verbose=verbose, pbar=pbar)
                continue

            # get the angle
            angle = angle_data.iloc[0]['azimuth']

            # rotate the image and the mask
            image = ri.rotate_image(image, angle, catch=catch, verbose=False, pbar=pbar)
            mask = ri.rotate_image(mask, angle, catch=catch, verbose=False, pbar=pbar)

        # if this image is geo-referenced, and we don't have tps, we need to get them
        if prev_footprint_exact is not None and prev_tps.shape[0] == 0:

            # load the georeferenced image
            ref_path = ref_fld + prev_id + ".tif"
            prev_image, prev_transform = liff.load_image_from_file(ref_path, return_transform=True)

            # we need an image, otherwise we cannot find tps
            if prev_image is not None:

                # we need to load the unreferenced mask (for mask creation)
                prev_image_orig = liff.load_image_from_file(prev_id)

                # create mask for this unreferenced image
                prev_mask = np.ones_like(prev_image_orig)

                # get the text boxes
                sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id='{prev_id}'"
                prev_m_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                prev_boxes = prev_m_data['text_bbox'].iloc[0].split(";")

                # Convert string representation to list of tuples
                prev_pairs = [tuple(map(int, coord[1:-1].split(','))) for coord in prev_boxes]

                # Set the regions in the array to 0
                for top_left_x, top_left_y, bottom_right_x, bottom_right_y in prev_pairs:
                    # Apply the transform to the tuple of coordinates
                    prev_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

                # remove the borders
                prev_mask = rb.remove_borders(prev_mask, prev_id)

                # get the angle
                sql_string = f"SELECT azimuth FROM images WHERE image_id='{prev_id}'"
                prev_m_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                prev_angle = prev_m_data['azimuth'].iloc[0]

                prev_mask = ri.rotate_image(prev_mask, prev_angle, start_angle=0)

                # transform the mask
                # Open a new in-memory file for writing.
                with MemoryFile() as memfile:
                    with memfile.open(driver='GTiff',
                                      height=prev_image.shape[0], width=prev_image.shape[1],
                                      count=1, dtype=str(prev_mask.dtype),
                                      crs='EPSG:3031', transform=prev_transform) as dataset:
                        # Write the mask to the in-memory file.
                        dataset.write(prev_mask, 1)

                    # Open the in-memory file for reading.
                    with memfile.open() as dataset:
                        # Now, dataset is a rasterio dataset that you can use as you would
                        # use a dataset opened from a file on disk.

                        prev_mask = dataset.read()[0]

                # find tps between the images
                prev_tps, prev_conf = ftp.find_tie_points(image, prev_image,  # noqa
                                                          mask_1=mask, mask_2=prev_mask,  # noqa
                                                          min_threshold=min_quality,
                                                          catch=catch, verbose=verbose, pbar=pbar)

                if debug_show_tps and prev_tps.shape[0] > 0:
                    dt.display_tiepoints([image, prev_image], points=prev_tps, confidences=prev_conf)

        # if this image is geo-referenced, and we don't have tps, we need to get them
        if next_footprint_exact is not None and next_tps.shape[0] == 0:

            # load the image
            ref_path = ref_fld + next_id + ".tif"

            next_image, next_transform = liff.load_image_from_file(ref_path, return_transform=True,
                                                                   catch=True, verbose=verbose, pbar=pbar)

            # we need an image, otherwise we cannot find tps
            if next_image is not None:

                # we need to load the unreferenced mask (for mask creation)
                next_image_orig = liff.load_image_from_file(next_id)

                # create mask for this unreferenced image
                next_mask = np.ones_like(next_image_orig)

                # get the text boxes
                sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id='{next_id}'"
                next_m_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                next_boxes = next_m_data['text_bbox'].iloc[0].split(";")

                # Convert string representation to list of tuples
                next_pairs = [tuple(map(int, coord[1:-1].split(','))) for coord in next_boxes]

                # Set the regions in the array to 0
                for top_left_x, top_left_y, bottom_right_x, bottom_right_y in next_pairs:
                    # Apply the transform to the tuple of coordinates
                    next_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

                # remove the borders
                next_mask = rb.remove_borders(next_mask, next_id)

                # get the angle
                sql_string = f"SELECT azimuth FROM images WHERE image_id='{next_id}'"
                next_m_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                next_angle = next_m_data['azimuth'].iloc[0]

                # rotate mask
                next_mask = ri.rotate_image(next_mask, next_angle, start_angle=0)

                # transform the mask
                # Open a new in-memory file for writing.
                with MemoryFile() as memfile:
                    with memfile.open(driver='GTiff',
                                      height=next_image.shape[0], width=next_image.shape[1],
                                      count=1, dtype=str(next_mask.dtype),
                                      crs='EPSG:3031', transform=next_transform) as dataset:
                        # Write the mask to the in-memory file.
                        dataset.write(next_mask, 1)

                    # Open the in-memory file for reading.
                    with memfile.open() as dataset:
                        # Now, dataset is a rasterio dataset that you can use as you would
                        # use a dataset opened from a file on disk.

                        next_mask = dataset.read()[0]

                # find tps between the images
                next_tps, next_conf = ftp.find_tie_points(image, next_image,
                                                          mask_1=mask, mask_2=next_mask,
                                                          min_threshold=min_quality,
                                                          catch=catch, verbose=verbose, pbar=pbar)

                if debug_show_tps and next_tps.shape[0] > 0:
                    dt.display_tiepoints([image, next_image], points=next_tps, confidences=next_conf)

        # this array will later contain all tie points
        all_tps = np.empty([0, 4])

        # check if we can use the prev image for gcps
        if prev_footprint_exact is not None and prev_tps.shape[0] > 0:

            # apply transformation to the tie-points of the prev image, so that these tps are absolute
            prev_tps[:, 2:] = np.array(
                rasterio.transform.xy(prev_transform, prev_tps[:, 3], prev_tps[:, 2])).transpose()

            # save these tps
            if debug_save_tps_shape:
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

        # check if we can use the next image for gcps
        if next_footprint_exact is not None and next_tps.shape[0] > 0:

            # apply transformation to the tie-points of the next image, so that these tps are absolute
            next_tps[:, 2:] = np.array(
                rasterio.transform.xy(next_transform, next_tps[:, 3], next_tps[:, 2])).transpose()

            # save these tps
            if debug_save_tps_shape:
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

        if debug_show_masks:
            if next_image is None:
                next_image = np.zeros([2,2])
                next_mask = np.zeros([2,2])
            if prev_image is None:
                prev_image = np.zeros([2,2])
                prev_mask = np.zeros([2,2])

            di.display_images([image, prev_image, next_image, mask, prev_mask, next_mask])

        if debug_group_tps:
            # Group rows with identical columns 3 and 4
            grouped_rows = np.split(all_tps, np.where(all_tps[:-1, :2] != all_tps[1:, :2])[0] + 1)

            # List to store the merged rows
            merged_rows = []

            # Iterate over the grouped rows
            for group in grouped_rows:

                # sometimes we have empty groups (?), easiest is to catch them
                if group.shape[0] == 0:
                    continue

                # Check the maximum difference in columns 3 and 4
                diff_3 = np.max(group[:, 2]) - np.min(group[:, 2])
                diff_4 = np.max(group[:, 3]) - np.min(group[:, 3])

                if diff_3 <= 10 and diff_4 <= 10:
                    # If the difference is within the threshold, compute the mean of columns 3 and 4
                    avg_values = np.mean(group[:, 2:], axis=0)
                    merged_rows.append(np.concatenate((group[0, :2], avg_values)))
                else:
                    # If the difference is too high, exclude all rows in the group
                    pass

            # Convert the merged rows to a NumPy array
            all_tps = np.array(merged_rows).astype(int)

            if all_tps.shape[0] == 0:
                p.print_v(f"There are no tps for {image_id}", verbose=verbose, pbar=pbar, color="red")
                continue

            # Group rows with identical columns 1 and 2
            grouped_rows = np.split(all_tps, np.where(all_tps[:-1, :2] != all_tps[1:, :2])[0] + 1)

            # List to store the merged rows
            merged_rows = []

            # Iterate over the grouped rows
            for group in grouped_rows:

                # sometimes we have empty groups (?), easiest is to catch them
                if group.shape[0] == 0:
                    continue

                # Check the maximum difference in columns 3 and 4
                diff_3 = np.max(group[:, 2]) - np.min(group[:, 2])
                diff_4 = np.max(group[:, 3]) - np.min(group[:, 3])

                if diff_3 <= 10 and diff_4 <= 10:
                    # If the difference is within the threshold, compute the mean of columns 3 and 4
                    avg_values = np.mean(group[:, 2:], axis=0)
                    merged_rows.append(np.concatenate((group[0, :2], avg_values)))
                else:
                    # If the difference is too high, exclude all rows in the group
                    pass

            # Convert the merged rows to a NumPy array
            all_tps = np.array(merged_rows).astype(int)

            # Sort the array based on columns 3 and 4
            sorted_tps = all_tps[np.lexsort((all_tps[:, 3], all_tps[:, 2]))]

            # Group rows with identical columns 3 and 4
            grouped_rows = np.split(sorted_tps, np.where(sorted_tps[:-1, 2:] != sorted_tps[1:, 2:])[0] + 1)

            # List to store the merged rows
            merged_rows = []

            # Iterate over the grouped rows
            for group in grouped_rows:

                # Sometimes we have empty groups, so we can skip them
                if group.shape[0] == 0:
                    continue

                # Check the maximum difference in columns 1 and 2
                diff_1 = np.max(group[:, 0]) - np.min(group[:, 0])
                diff_2 = np.max(group[:, 1]) - np.min(group[:, 1])

                if diff_1 <= 10 and diff_2 <= 10:
                    # If the difference is within the threshold, compute the mean of columns 1 and 2
                    avg_values = np.mean(group[:, :2], axis=0)
                    merged_rows.append(np.concatenate((avg_values, group[0, 2:],)))
                else:
                    # If the difference is too high, exclude all rows in the group
                    pass

            # Convert the merged rows to a NumPy array and convert to integer data type
            all_tps = np.array(merged_rows).astype(int)

        # filter for outliers
        if debug_filter_outliers and all_tps.shape[0] >= 4:
            _, mask = cv2.findHomography(all_tps[:, 0:2], all_tps[:, 2:4], cv2.RANSAC, 5.0)
            mask = mask.flatten()

            p.print_v(f"{np.count_nonzero(mask)} outliers removed.", verbose=verbose, pbar=pbar)

            # 1 means outlier
            all_tps = all_tps[mask == 0]

        output_folder = "/data_1/ATM/data_1/playground/georef2/tiffs/georeferenced1"
        transform_method = "rasterio"

        if all_tps.shape[0] < min_nr_tps:
            p.print_v(f"{image_id} has too few tie-points ({all_tps.shape[0]} < {min_nr_tps})",
                      verbose=verbose, color="red", pbar=pbar)
            return

        # switch satellite tps and image tps
        tps_switched = np.concatenate((all_tps[:, 2:4], all_tps[:, 0:2]), axis=1)

        if debug_save_tps_shape:
            path_shape_file = "/data_1/ATM/data_1/playground/georef2/shape_files/tie_points_1/" \
                              + image_id + "_points.shp"

            # Create a DataFrame from the NumPy array
            data = {'x': tps_switched[:, 0], 'y': tps_switched[:, 1],
                    'x_pic': tps_switched[:, 2], 'y_pic': tps_switched[:, 3]
                    }
            df = pd.DataFrame(data)

            geometry = gpd.points_from_xy(x=df['x'], y=df['y'])
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:3031')

            gdf.to_file(path_shape_file, driver='ESRI Shapefile')

        # should we check the images after geo-referencing?
        if debug_final_check:

            good_image = cgi.check_georef_validity(path_georef_tiffs + "/" + image_id + ".tif")

            # remove image from the folders
            if good_image is False:
                try:
                    os.remove(path_georef_tiffs + "/" + image_id + ".tif")
                    os.remove(base_path + f"/images/tie_points/{image_id}.png")
                    os.remove(base_path + "/shape_files/tie_points/" + image_id + "_points.shp")
                except:
                    pass

        if debug_save_georef_tiffs:
            transform = ag.apply_gpcs(image_id, image, tps_switched, output_folder,
                                   transform_method, gdal_order=1,
                                   catch=True)

            if transform is not None:
                p.print_v(f"{image_id} successfully georeferenced",
                            color="green", verbose=verbose, pbar=pbar)


if __name__ == "__main__":
    _sql_string = "SELECT image_id FROM images WHERE view_direction = 'V'"
    ids = ctd.get_data_from_db(_sql_string).values.tolist()
    ids = [item for sublist in ids for item in sublist]

    import random

    random.shuffle(ids)

    ids = ["CA212332V0042"]

    georef3(ids, catch=_catch, verbose=_verbose)
