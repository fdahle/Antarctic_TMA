import copy
import os.path

import numpy as np
import pandas as pd

from shapely import wkt

import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.rotate_image as ri

import display.display_images as di
import display.display_tiepoints as dt

import image_georeferencing.sub.adjust_image_resolution as air
import image_georeferencing.sub.buffer_footprint as bf
import image_georeferencing.sub.calc_footprint_approx as cfa
import image_georeferencing.sub.load_satellite_data as lsd
import image_georeferencing.old.update_table_images_footprints as utif

_path_ortho = ""
# img_ids = ["CA213732V0032", "CA213732V0033", "CA213732V0034"]
# img_ids = ["CA214732V0029", "CA214732V0030", "CA214732V0031", "CA214732V0032", "CA214732V0033"]
img_ids = []

debug_save_approx_footprint_avg = True

bool_locate_image_in_area = False
bool_use_sat_images_with_date = False

save_path = "/data_1/ATM/data_1/playground/georeference/images/"
save_path = None

def georef_test(image_ids,
                use_avg_vals=False,
                overwrite=False, catch=True, verbose=False):
    p.print_v(f"Start: georeference_all_images ({len(image_ids)} images)")

    # convert the list of image_ids to a string
    str_image_ids = "('" + "', '".join(image_ids) + "')"

    # get information from images
    sql_string = f"SELECT * FROM images WHERE image_id IN {str_image_ids}"
    data_images = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)

    if data_images is None:
        print("Data images failed")
        return

    # get information we need from image_extracted
    sql_string = f"SELECT image_id, ST_AsText(footprint_approx) AS footprint_approx, " \
                 f"height, focal_length " \
                 f"FROM images_extracted WHERE image_id IN {str_image_ids}"
    data_extracted = ctd.get_data_from_db(sql_string, catch=False, verbose=verbose)

    if data_extracted is None:
        print("Data extracted failed")
        return

    # merge the information
    data = pd.merge(data_images, data_extracted, on='image_id')

    # first iterate all images to get the approx_footprint and angles
    #for image_id in (pbar := tqdm(image_ids)):
    counter = 0
    for image_id in image_ids:
        counter = counter + 1
        pbar=None

        # only geo-reference vertical images
        if "V" not in image_id:
            continue

        if overwrite is False and save_path is not None and \
                os.path.isfile(save_path + "{image_id}.png"):
            continue

        p.print_v(f"georef {image_id} ({counter}/{len(image_ids)})", color="black")

        # get the row from the dataframe
        row = data.loc[data['image_id'] == image_id]

        if row.empty:
            if catch:
                p.print_v(f"Warning: Information missing in dataframe for '{image_id}'", verbose, pbar=pbar)
                continue
            else:
                raise ValueError(f"Information missing in dataframe for '{image_id}'")
        else:
            # convert dataframe to dict
            row = row.iloc[0]

        # get the interesting attributes from the row
        footprint_approx = row["footprint_approx"]
        angle = row["azimuth"]
        view_direction = row["view_direction"]
        focal_length = row["focal_length"]
        height = row["height"]
        month = row["date_month"]
        flight_path = int(row["tma_number"])

        bool_save_footprint = True

        # if we don't have a focal length for this image, we must skip this image
        if focal_length is None:
            if use_avg_vals:
                bool_save_footprint = False
                focal_length = 152.457

            else:
                if catch:
                    p.print_v(f"Warning: Focal length missing for '{image_id}'",
                              verbose, pbar=pbar)
                    continue
                else:
                    raise ValueError(f"Warning: Focal length missing for '{image_id}'")

        # if we don't have a height for this image, we must skip this image
        if height is None or np.isnan(height):
            if use_avg_vals:
                bool_save_footprint = False
                height = 25000
            else:
                if catch:
                    p.print_v(f"Warning: Height missing for '{image_id}'",
                              verbose, pbar=pbar)
                    continue
                else:
                    raise ValueError(f"Warning: Height missing for '{image_id}'")

        # if we don't have an angle for this image, we must skip this image
        if angle is None:
            if catch:
                p.print_v(f"Warning: Angle missing for '{image_id}'",
                          verbose, pbar=pbar)
                continue
            else:
                raise ValueError(f"Warning: Angle missing for '{image_id}'")

        footprint_approx = None

        # if the approx_footprint is not existing yet, we need to calculate it
        if footprint_approx is None:
            # get the shapefile position information
            x = row["x_coords"]
            y = row["y_coords"]

            # calculate an approximate approx_footprint based on the information we have from the shapefile
            footprint_approx = cfa.calc_footprint_approx(x, y, angle, view_direction,
                                                         height, focal_length,
                                                         catch=True, verbose=False, pbar=pbar)

            if footprint_approx is None:
                p.print_v(f"Failed: geo_ref_test (no approximate footprint for {image_id})",
                      verbose=verbose, pbar=pbar)
                continue

            # save this approx_footprint in the database, but not if we use avg values
            if debug_save_approx_footprint_avg or bool_save_footprint:
                success = utif.update_table_images_footprints(image_id, "footprint_approx", footprint_approx,
                                                              overwrite=overwrite, catch=False, verbose=False, pbar=pbar)

                if success is None:
                    print(f"Update table images footprint failed for {image_id}")
                    continue
        else:
            footprint_approx = wkt.loads(footprint_approx)

        # did we already have other pictures of this flight from which we can use some information
        sql_string = "SELECT image_id, footprint_corrected FROM images_extracted " \
                     "WHERE footprint_corrects is NOT NULL AND " \
                     f"SUBSTRING(image_id, 3, 7) = '{flight_path}'"

        # buffer the approx_footprint & get bounds for satellite
        footprint_buffered = bf.buffer_footprint(footprint_approx, buffer_val=2000,
                                                 catch=False, verbose=False, pbar=pbar)

        if footprint_buffered is None:
            continue

        # get the satellite bounds from this approx_footprint
        sat_bounds = list(footprint_buffered.bounds)
        sat_bounds = [int(x) for x in sat_bounds]

        if bool_use_sat_images_with_date is False:
            month = None

        # get the satellite image of the approx_footprint
        sat_image, sat_transform = lsd.load_satellite_data(sat_bounds, month=month,
                                                           fallback_month=True,
                                                           return_transform=True,
                                                           catch=catch, verbose=False)

        if sat_image is None:
            p.print_v(f"No satellite image for {image_id}", verbose=verbose, pbar=pbar)
            continue

        # load image and rotate
        img = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)

        if img is None:
            print(f"No image for {image_id}", verbose=verbose, pbar=pbar)
            continue

        # adjust the image so that pixel size is similar to satellite image
        img_adjusted, y_factor, x_factor = air.adjust_image_resolution(sat_image, sat_bounds,
                                                                       img, footprint_approx.bounds)

        # remove the borders
        img_no_borders, edge_dims = rb.remove_borders(img, image_id=image_id, return_edge_dims=True,
                                                      catch=catch, verbose=False, pbar=pbar)
        img_no_borders = ri.rotate_image(img_no_borders, angle, start_angle=90)

        import image_extraction.calc_image_complexity as cic
        complexity = cic.calc_image_complexity(img_no_borders)

        if complexity < 0.3:
            print(f"Image is too simple ({complexity})")
            continue

        # adapt the border dimension to account for smaller image
        change_y = img.shape[0] / img_adjusted.shape[0]
        change_x = img.shape[1] / img_adjusted.shape[1]
        if change_y > 0 or change_x > 0:
            change_y = 1 / change_y
            change_x = 1 / change_x
        edge_dims[0] = int(edge_dims[0] * change_x)
        edge_dims[1] = int(edge_dims[1] * change_x)
        edge_dims[2] = int(edge_dims[2] * change_y)
        edge_dims[3] = int(edge_dims[3] * change_y)

        # remove the border from the small image and then rotate
        img_adjusted = img_adjusted[edge_dims[2]:edge_dims[3], edge_dims[0]:edge_dims[1]]
        img_adjusted = ri.rotate_image(img_adjusted, angle, start_angle=0)

        import image_tie_points.find_tie_points as ftp
        import image_georeferencing.sub.find_footprint_direction as ffd

        di.display_images([sat_image, img_adjusted], title=f"{image_id}: start")


        # try to locate the image at the position of the approx approx_footprint(and the surroundings)
        if bool_locate_image_in_area:


            # set the params
            overlap = 1 / 3  # how much overlap do we want to have
            max_order = 3  # what is the maximum order we're checking
            success_criteria = "num_points"  # What is a success? 'num_points' or 'avg_conf'

            # get the width and height of the satellite image
            sat_width = sat_image.shape[1] * np.abs(sat_transform[0])
            sat_height = sat_image.shape[2] * np.abs(sat_transform[4])

            # calculate the step size
            step_x = int((1 - overlap) * sat_width)
            step_y = int((1 - overlap) * sat_height)

            # track what we did check already
            lst_checked_combinations = []

            # define in which rotations we want to check
            rotations = [0, 90, 180, 270]

            # iterate the orders (start with the lowest first)
            for order in range(max_order):

                # create a combinations dict
                combinations = []
                for i in range(-order * step_x, (order + 1) * step_x, step_x):
                    for j in range(-order * step_y, (order + 1) * step_y, step_y):
                        combinations.append([i, j])

                # per combination, we want to check our performance
                best_combination = None
                best_nr_of_points = 0
                best_avg_conf = 0

                # check all combinations
                for combination in combinations:

                    # copy the original sat bounds (to not change them)
                    adapted_sat_bounds = copy.deepcopy(sat_bounds)

                    # first adapt x
                    adapted_sat_bounds[0] = adapted_sat_bounds[0] + combination[0]
                    adapted_sat_bounds[2] = adapted_sat_bounds[2] + combination[0]

                    # and then y
                    adapted_sat_bounds[1] = adapted_sat_bounds[1] + combination[1]
                    adapted_sat_bounds[3] = adapted_sat_bounds[3] + combination[1]

                    sat_image, sat_transform = lsd.load_satellite_data(adapted_sat_bounds,
                                                                       return_transform=True,
                                                                       catch=catch, verbose=False)

                    # another loop to check all 4 rotations
                    for rotation in rotations:

                        combination_str = str(combination[0]) + ";" + str(combination[1]) + ";" + \
                                          str(rotation)

                        # we don't need to check again
                        if combination_str in lst_checked_combinations:
                            continue
                        lst_checked_combinations.append(combination_str)

                        print(f"check {combination_str}")

                        # copy the image to not change it and then rotate it
                        img_rotated = copy.deepcopy(img_adjusted)
                        img_rotated = ri.rotate_image(img_rotated, angle, start_angle=rotation)

                        if sat_image is None or img_rotated is None:
                            continue

                        # find the tie-points
                        points_all, conf_all = ftp.find_tie_points(sat_image, img_rotated,
                                                                   mask_1=None, mask_2=None,
                                                                   min_threshold=0.2,
                                                                   extra_mask_padding=10,
                                                                   additional_matching=True,
                                                                   extra_matching=True,
                                                                   keep_resized_points=True,
                                                                   keep_additional_points=True,
                                                                   catch=catch, verbose=False, pbar=pbar)

                        if points_all is None:
                            nr_points_all = 0
                        else:
                            nr_points_all = points_all.shape[0]

                        print(nr_points_all)

                        # get the average conf
                        if conf_all is None:
                            avg_conf = 0
                        else:
                            avg_conf = np.mean(np.asarray(conf_all))
                            if np.isnan(avg_conf):
                                avg_conf = 0

                        if success_criteria == "num_points" and nr_points_all > best_nr_of_points:
                            best_combination = combination_str
                            best_nr_of_points = nr_points_all
                            best_avg_conf = avg_conf
                        elif success_criteria == "avg_conf" and avg_conf > best_avg_conf:
                            best_combination = combination_str
                            best_nr_of_points = nr_points_all
                            best_avg_conf = avg_conf

                # do we need another order?
                if best_nr_of_points > 25:
                    break

            if best_nr_of_points > 25:  # noqa
                col = "green"
            else:
                col = "red"

            p.print_v(best_combination, color=col)  # noqa
            p.print_v(best_nr_of_points, color=col)  # noqa
            p.print_v(best_combination, color=col)  # noqa

            best_params = best_combination.split(";")

            # now get the final tie-points again for the best combination
            # copy the original sat bounds (to not change them)
            adapted_sat_bounds = copy.deepcopy(sat_bounds)

            # first adapt x
            adapted_sat_bounds[0] = adapted_sat_bounds[0] + int(best_params[0])
            adapted_sat_bounds[2] = adapted_sat_bounds[2] + int(best_params[0])

            # and then y
            adapted_sat_bounds[1] = adapted_sat_bounds[1] + int(best_params[1])
            adapted_sat_bounds[3] = adapted_sat_bounds[3] + int(best_params[1])

            sat_image, sat_transform = lsd.load_satellite_data(adapted_sat_bounds,
                                                               month=month, fallback_month=True,
                                                               return_transform=True,
                                                               catch=catch, verbose=False)

            sat_bounds = adapted_sat_bounds

            # copy the image to not change it and then rotate it to the best params
            img_rotated = copy.deepcopy(img_adjusted)
            img_rotated = ri.rotate_image(img_rotated, int(best_params[2]))

            # find the tie-points
            points_all, conf_all = ftp.find_tie_points(sat_image, img_rotated,
                                                       mask_1=None, mask_2=None,
                                                       min_threshold=0.2,
                                                       extra_mask_padding=10,
                                                       additional_matching=True,
                                                       extra_matching=True,
                                                       keep_resized_points=True,
                                                       keep_additional_points=True,
                                                       catch=catch, verbose=False, pbar=pbar)

            dt.display_tiepoints([sat_image, img_rotated], points=points_all, title=f"{image_id} after combination: {best_combination}")

        # some parameters required for the tweaking of the approx_footprint
        num_points = 0
        avg_conf = 0
        sat_bounds_old = None
        points_all_old = None
        conf_all_old = None

        counter = 0

        while True:

            counter = counter + 1
            if counter == 10:
                break

            #di.display_images([sat_image, img_adjusted], title=f"Counter {counter}")

            points_all, conf_all = ftp.find_tie_points(sat_image, img_adjusted,
                                                       mask_1=None, mask_2=None,
                                                       min_threshold=0.2,
                                                       extra_mask_padding=10,
                                                       additional_matching=True,
                                                       extra_matching=True,
                                                       keep_resized_points=True,
                                                       keep_additional_points=True,
                                                       catch=catch, verbose=False, pbar=pbar)

            dt.display_tiepoints([sat_image, img_adjusted], points_all, conf_all)

            exit()

            # sad, we didn't find any points
            if points_all.shape[0] == 0:
                p.print_v(f"No tie-points found for {image_id}", verbose=verbose, pbar=pbar)
                break

            # if the number of points is going down we don't need to continue but get the old values again
            if num_points >= points_all.shape[0]:
                print(f"Points going down ({points_all.shape} < {num_points}")
                sat_bounds = sat_bounds_old
                sat_image = sat_image_old
                points_all = points_all_old
                conf_all = conf_all_old
                break

            # save the values so that we can roll back
            num_points = points_all.shape[0]
            sat_image_old = copy.deepcopy(sat_image)
            sat_bounds_old = copy.deepcopy(sat_bounds)
            points_all_old = copy.deepcopy(points_all)
            conf_all_old = copy.deepcopy(conf_all)

            # find the direction in which we need to change the satellite image
            step_y, step_x = ffd.find_footprint_direction(img_adjusted, points_all[:, 2:4], conf_all)

            step_y = step_y * -1

            print(step_y, step_x)

            # adapt the satellite bounds
            sat_bounds[0] = sat_bounds[0] + step_x
            sat_bounds[1] = sat_bounds[1] + step_y
            sat_bounds[2] = sat_bounds[2] + step_x
            sat_bounds[3] = sat_bounds[3] + step_y

            # get the adapted satellite image of the approx_footprint
            sat_image, sat_transform = lsd.load_satellite_data(sat_bounds,
                                                               month=month, fallback_month=True,
                                                               return_transform=True,
                                                               catch=True, verbose=verbose)

            if sat_image is None:
                print("Failed: adapted satellite image")
                return

        if points_all.shape[0] < 4:
            p.print_v(f"Failed: georef_test (too few tie-points for {image_id})", verbose=verbose, pbar=pbar)
            continue

        dt.display_tiepoints([sat_image, img_rotated], points=points_all,
                             title=f"{image_id}: after tweaking")

        # filter outliers
        import cv2
        _, mask = cv2.findHomography(points_all[:, 0:2], points_all[:, 2:4], cv2.RANSAC, 5.0)
        points_all = points_all[mask.flatten() == 0]
        conf_all = conf_all[mask.flatten() == 0]
        print(f"{np.count_nonzero(mask.flatten())} entries are removed as outliers")

        if points_all.shape[0] < 4:
            print("Too few points found after outlier removal")
            continue

        if save_path is None:
            dt.display_tiepoints([sat_image, img_adjusted], points=points_all, title=f"{image_id}")
        else:
            dt.display_tiepoints([sat_image, img_adjusted], points=points_all, title=f"{image_id}",
                                 save_path=save_path+ f"/{image_id}_tps.png")

        # project satellite points to absolute coords
        points_all[:, 0] = points_all[:, 0] * sat_transform[0]
        points_all[:, 1] = points_all[:, 1] * sat_transform[4]
        points_all[:, 0] = points_all[:, 0] + sat_transform[2]
        points_all[:, 1] = sat_transform[5] - sat_image.shape[0] * sat_transform[4] + points_all[:, 1]

        # now project back the tie-points to an un-adjusted image
        # points_all[:2] = points_all[:2] + edge_dims[0]
        # points_all[:3] = points_all[:3] + edge_dims[2]

        points_all[:, 2] = points_all[:, 2] * 1 / x_factor
        points_all[:, 3] = points_all[:, 3] * 1 / y_factor

        abs_points_subset = points_all

        import math
        nr_divisons = int(math.sqrt(num_points))

        subset_height = img_no_borders.shape[0] / nr_divisons
        subset_width = img_no_borders.shape[1] / nr_divisons

        gcp_idx = []
        for y in range(nr_divisons):
            for x in range(nr_divisons):

                min_x = int(x * subset_width)
                max_x = int((x + 1) * subset_width)
                min_y = int(y * subset_height)
                max_y = int((y + 1) * subset_height)

                # get the indices of the points of this subset
                sub_idx = np.where((points_all[:, 2] > min_x) & (points_all[:, 2] <= max_x) &
                                   (points_all[:, 3] > min_y) & (points_all[:, 3] <= max_y))
                sub_idx = sub_idx[0]

                # if there are no points we can skip this iteration
                if len(sub_idx) == 0:
                    continue

                # find the index with the maximum confidence of this subset
                idx_max_conf = sub_idx[np.argmax(conf_all[sub_idx])]

                gcp_idx.append(idx_max_conf)

        # get the subset of points
        points_all = points_all[gcp_idx, :]  # noqa
        conf_all = conf_all[gcp_idx]

        # check if the points are still valid
        # if np.sum(abs_points_subset[:,0] > geo_ref_bounds[2]) > 0:
        #    print(f"WARNING: {np.sum(abs_points_subset[:,0] > geo_ref_bounds[2])} x values are too large")
        # if np.sum(abs_points_subset[:,0] < geo_ref_bounds[0]) > 0:
        #    print(f"WARNING: {np.sum(abs_points_subset[:,0] < geo_ref_bounds[0])} x values are too small")
        # if np.sum(abs_points_subset[:,1] > geo_ref_bounds[3]) > 0:
        #    print(f"WARNING: {np.sum(abs_points_subset[:,1] > geo_ref_bounds[3])} y values are too large")
        # if np.sum(abs_points_subset[:,1] < geo_ref_bounds[1]) > 0:
        #    print(f"WARNING: {np.sum(abs_points_subset[:,1] < geo_ref_bounds[1])} y values are too small")

        import geopandas as gpd
        transform_method = "gdal"
        from osgeo import gdal, osr
        import rasterio
        gdal_order = 1

        output_path = "/data_1/ATM/data_1/playground/georeference/"

        # Create a GeoDataFrame to save tie points
        debug_save_gcps = True
        if debug_save_gcps:
            tmp_points_path = output_path + "tie_points/" + image_id + "_points.shp"
            geometry = gpd.points_from_xy(x=abs_points_subset[:, 0], y=abs_points_subset[:, 1])
            gdf = gpd.GeoDataFrame(geometry=geometry, crs='epsg:3031')
            gdf.to_file(tmp_points_path, driver='ESRI Shapefile')

        gcps = []

        # create the ground control points
        for i in range(abs_points_subset.shape[0]):
            row = abs_points_subset[i, :]

            gcp_coords = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))

            if transform_method == "gdal":
                gcp = gdal.GCP(gcp_coords[0], gcp_coords[1], 0, gcp_coords[2], gcp_coords[3])
            elif transform_method == "rasterio":
                gcp = rasterio.control.GroundControlPoint(gcp_coords[3], gcp_coords[2], gcp_coords[0],
                                                          gcp_coords[1])
            gcps.append(gcp)

        from rasterio.transform import from_gcps

        if transform_method == "gdal":

            # create the tiff file in memory
            driver = gdal.GetDriverByName("MEM")
            ds = driver.Create("", img_no_borders.shape[1], img_no_borders.shape[0], 1, gdal.GDT_Byte)
            for i in range(1):
                band = ds.GetRasterBand(i + 1)
                band.WriteArray(img_no_borders)
                band.SetNoDataValue(0)

            # Set the GCPs for the image
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3031)
            ds.SetGCPs(gcps, srs.ExportToWkt())

            warp_options = ["-r", "near", "-et", "0"]  # , "-refine_gcps"]
            if gdal_order in [1, 2, 3]:
                warp_options.append("-order")
                warp_options.append(str(gdal_order))

            # save the new geo-referenced tiff-file
            output_path_tiffs = output_path + "tiffs/" + image_id + ".tif"
            output_ds = gdal.Warp(output_path_tiffs, ds,  # noqa
                                  dstSRS=srs.ExportToWkt(), options=warp_options)
            output_ds.GetRasterBand(1).SetMetadataItem("COMPRESSION", "LZW")
            output_ds.FlushCache()
            output_ds = None

        elif transform_method == "rasterio":
            transform = from_gcps(gcps)

            profile = {
                'driver': 'GTiff',
                'height': img_no_borders.shape[0],
                'width': img_no_borders.shape[1],
                'count': 1,
                'dtype': img_no_borders.dtype,
                'crs': 'EPSG:3031',
                'transform': transform
            }

            with rasterio.open("/data_1/ATM/data_1/playground/tiffs/" + image_id + ".tif", "w", **profile) as dst:
                dst.write(img_no_borders, 1)

            p.print_v(f"{image_id} successfully georeferenced!", color="green")

        p.print_v("Finished: georef_test", verbose=verbose, pbar=pbar)
        print(f"georef finished")


if __name__ == "__main__":

    if len(img_ids) == 0:
        import base.connect_to_db as ctd

        sql_string = "SELECT image_id FROM images_extracted WHERE complexity > 0.8"
        data = ctd.get_data_from_db(sql_string)
        img_ids = data['image_id'].tolist()

    #random.shuffle(img_ids)

    img_ids = ["CA214732V0032"]

    georef_test(img_ids, use_avg_vals=True, overwrite=True, catch=True, verbose=True)
