import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.io import MemoryFile

import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.convert_image_to_footprint as citf
import image_georeferencing.sub.filter_tp_outliers as fto

import image_tie_points.find_tie_points as ftp

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.rotate_image as ri

import display.display_images as di
import display.display_tiepoints as dt

debug_show_images = False
debug_show_tie_points_unfiltered = False
debug_show_tie_points_filtered = False
debug_show_tie_points_final = True

# some other debug params
debug_save = True


def georef_img(image_id, path_fld, path_georef_tiffs,
               max_distance_neighbours=2,
               min_tps_final=25,
               mask_images=True,
               min_threshold=0.8,
               enhance_image=True, enhancement_min=0.02, enhancement_max=0.98,
               filter_tps=True, filter_use_ransac=True, filter_ransac_val=10,
               filter_use_angle=True, filter_angle=5,
               transform_method="rasterio", transform_order=3,
               save_tie_points=True,
               catch=True, verbose=False, pbar=None):
    """
    georef_img():
    geo-reference an image based on adjacent images

    Args:
        image_id (String): The id of the image we want to geo-reference
        path_fld (String): Where do we want to store the temporary tif, png & shapefile
        path_georef_tiffs (String): path to the already geo-referenced tiff-files stored
        min_tps_final (int, 25): We need at least that many tps to geo-reference
        mask_images (Boolean, true): Should masks be applied when looking for matches between images
        min_threshold (Float, 0.8): The min confidence of tie-points that are used for geo-referencing
        filter_tps (Boolean, true): If true, tps are checked for outliers
        filter_use_ransac (Boolean, true): If true, ransac is used for outlier filtering
        filter_ransac_val (int, 10): Maximum allowed re-projection error for ransac
        filter_use_angle (Boolean, true): If true, outliers are filtered with angles
        filter_angle (int, 5): Maximum allowed deviation for angle filtering
        transform_method (String, "rasterio"): Which library will be used for applying gcps ('rasterio' or 'gdal')
        transform_order (Int, 3): Which polynomial should be used for geo-referencing (only gdal)
        save_tie_points (Boolean, true): If true, we are saving information for tie-points in the tmp folder
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:


    """

    # define already the status
    status = ""

    # define where we save stuff
    path_tiff = path_fld + "/" + image_id + ".tif"
    path_tie_points_shape = path_fld + "/" + image_id + "_points.shp"

    # get the direct neighbours
    image_nr = int(image_id[-4:])
    prev_ids = [image_id[:-4] + '{:04d}'.format(int(image_nr) - i) for i in range(1, max_distance_neighbours + 1)]
    next_ids = [image_id[:-4] + '{:04d}'.format(int(image_nr) + i) for i in range(1, max_distance_neighbours + 1)]

    # convert ids to a string
    prev_ids_string = ",".join([f"'{_id}'" for _id in prev_ids])
    next_ids_string = ",".join([f"'{_id}'" for _id in next_ids])

    # get the data of the image
    sql_string_image = f"SELECT image_id, azimuth FROM images WHERE image_id='{image_id}'"
    data_image = ctd.get_data_from_db(sql_string_image, catch=True)

    # get the data of the image_extracted
    sql_string_extracted = f"SELECT * FROM images_extracted WHERE image_id='{image_id}'"
    data_extracted = ctd.get_data_from_db(sql_string_extracted, catch=True)

    # merge the data from both tables
    data = pd.merge(data_image, data_extracted, on='image_id')

    # check if getting data worked
    if data is None:
        p.print_v(f"Something went wrong getting data for {image_id}",
                  color="red", verbose=verbose, pbar=pbar)
        return_tuple = (None, None, "failed:get_data_from_db", None, None)
        return return_tuple

    # check if we have an entry for the image
    if data.shape[0] == 0:
        p.print_v(f"There is no entry for for {image_id}",
                  color="red", verbose=verbose, pbar=pbar)
        return_tuple = (None, None, "invalid:no_entry_for_image", None, None)
        return return_tuple
    else:
        data = data.iloc[0]

    # get the data of the neighbours
    prev_sql_string = f"SELECT image_id, footprint_exact, text_bbox FROM images_extracted " \
                      f"WHERE footprint_type='satellite' AND " \
                      f"image_id IN ({prev_ids_string})"
    next_sql_string = f"SELECT image_id, footprint_exact, text_bbox FROM images_extracted WHERE " \
                      f"footprint_type='satellite' AND " \
                      f"image_id IN ({next_ids_string})"
    prev_data = ctd.get_data_from_db(prev_sql_string, catch=True)
    next_data = ctd.get_data_from_db(next_sql_string, catch=True)

    # checked if getting data worked for prev
    if prev_data is None:
        p.print_v(f"Something went wrong getting data for prev_ids",
                  color="red", verbose=verbose, pbar=pbar)
        return_tuple = (None, None, "failed:get_data_from_db", None, None)
        return return_tuple

    # checked if getting data worked for next
    if next_data is None:
        p.print_v(f"Something went wrong getting data for next_ids",
                  color="red", verbose=verbose, pbar=pbar)
        return_tuple = (None, None, "failed:get_data_from_db", None, None)
        return return_tuple

    # check if we have an exact footprint for the prev neighbour
    if prev_data.shape[0] > 0:
        prev_ids = prev_data['image_id'].tolist()
        prev_footprints_exact = prev_data['footprint_exact'].tolist()
    else:
        prev_ids = []
        prev_footprints_exact = []

    # check if we have an exact footprint for the prev neighbour
    if next_data.shape[0] > 0:
        next_ids = next_data['image_id'].tolist()
        next_footprints_exact = next_data['footprint_exact'].tolist()
    else:
        next_ids = []
        next_footprints_exact = []

    # if both neighbours are not georeferenced -> skip this image
    if len(prev_footprints_exact) == 0 and len(next_footprints_exact) == 0:
        p.print_v("No direct neighbours are georeferenced", color="red", verbose=verbose, pbar=pbar)
        return_tuple = (None, None, "no_neighbours", None, None)
        return return_tuple

    # load base image
    image = liff.load_image_from_file(image_id, catch=catch)

    # save the images and transforms
    prev_images = []
    prev_transforms = []
    next_images = []
    next_transforms = []

    # load prev images
    for i, prev_footprint in enumerate(prev_footprints_exact):

        # init prev_image & transform already
        prev_image = None
        prev_transform = None

        if prev_footprint is not None:
            prev_path = path_georef_tiffs + "/" + prev_ids[i] + ".tif"
            prev_image, prev_transform = liff.load_image_from_file(prev_path, return_transform=True,
                                                                   catch=catch)

        if prev_footprint is None or prev_image is None:
            prev_image = (np.zeros(2, 2))
            prev_transform = None

        prev_images.append(prev_image)
        prev_transforms.append(prev_transform)

    # load next image
    for i, next_footprint in enumerate(next_footprints_exact):

        # init next_image & transform already
        next_image = None
        next_transform = None

        if next_footprint is not None:
            next_path = path_georef_tiffs + "/" + next_ids[i] + ".tif"
            next_image, next_transform = liff.load_image_from_file(next_path, return_transform=True,
                                                                   catch=catch)

        if next_footprint is None or next_image is None:
            next_image = (np.zeros(2, 2))
            next_transform = None

        next_images.append(next_image)
        next_transforms.append(next_transform)

    # get the borders of the image
    image_no_borders, border_dims = rb.remove_borders(image, image_id, return_edge_dims=True)

    # should we mask the images when looking for tps?
    if mask_images:

        # create mask and remove borders
        mask = np.ones_like(image)
        mask[:border_dims[2], :] = 0
        mask[border_dims[3]:, :] = 0
        mask[:, :border_dims[0]] = 0
        mask[:, border_dims[1]:] = 0

        # get the text boxes
        text_boxes = data['text_bbox'].split(";")

        # Convert string representation to list of tuples
        coordinate_pairs = [tuple(map(lambda x: int(float(x)), coord[1:-1].split(','))) for coord in text_boxes]

        # Set the regions in the array to 0
        for top_left_x, top_left_y, bottom_right_x, bottom_right_y in coordinate_pairs:
            mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

        # remove the edge from the mask
        mask = rb.remove_borders(mask, image_id)

        # the mask must be rotated
        mask = ri.rotate_image(mask, data['azimuth'], catch=catch, verbose=False, pbar=pbar)

    # the image must be rotated as well
    image = ri.rotate_image(image_no_borders, data['azimuth'], catch=catch, verbose=False, pbar=pbar)

    prev_masks = []
    next_masks = []

    if mask_images:

        # check for all prev footprints if it is possible to mask
        for i, prev_footprint in enumerate(prev_footprints_exact):

            # we only mask if there's a prev footprint
            if prev_footprint is not None:

                # we need the unreferenced image as well for mask creation
                prev_image_unreferenced = liff.load_image_from_file(prev_ids[i], catch=True,
                                                                    verbose=False, pbar=pbar)

                # create mask
                prev_mask_unreferenced = np.ones_like(prev_image_unreferenced)

                # get the text boxes
                prev_text_boxes = prev_data['text_bbox'].iloc[i].split(";")

                # Convert string representation to list of tuples
                prev_coordinate_pairs = [tuple(map(lambda x: int(float(x)), coord[1:-1].split(',')))
                                         for coord in prev_text_boxes]

                # Set the regions in the array to 0
                for top_left_x, top_left_y, bottom_right_x, bottom_right_y in prev_coordinate_pairs:
                    prev_mask_unreferenced[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

                # remove the borders
                prev_mask_unreferenced = rb.remove_borders(prev_mask_unreferenced, prev_ids[i])

                # get the angle of prev image
                sql_string = f"SELECT azimuth FROM images WHERE image_id='{prev_ids[i]}'"
                prev_azimuth_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                prev_angle = prev_azimuth_data['azimuth'].iloc[0]

                # rotate
                prev_mask_unreferenced = ri.rotate_image(prev_mask_unreferenced, prev_angle, start_angle=0)

                # transform the mask
                # Open a new in-memory file for writing.
                with MemoryFile() as memfile:
                    with memfile.open(driver='GTiff',
                                      height=prev_image.shape[0], width=prev_image.shape[1],  # noqa
                                      count=1, dtype=str(prev_mask_unreferenced.dtype),
                                      crs='EPSG:3031', transform=prev_transform) as dataset:  # noqa
                        # Write the mask to the in-memory file.
                        dataset.write(prev_mask_unreferenced, 1)

                    # Open the in-memory file for reading.
                    with memfile.open() as dataset:
                        # Now, dataset is a rasterio dataset that you can use as you would
                        # use a dataset opened from a file on disk.

                        prev_mask = dataset.read()[0]
            else:
                prev_mask = np.ones((2, 2))

            prev_masks.append(prev_mask)

        # check for all prev footprints if it is possible to mask
        for i, next_footprint in enumerate(next_footprints_exact):

            # we only mask if there's a prev footprint
            if next_footprint is not None:

                # we need the unreferenced image as well for mask creation
                next_image_unreferenced = liff.load_image_from_file(next_ids[i], catch=True,
                                                                    verbose=False, pbar=pbar)

                # create mask
                next_mask_unreferenced = np.ones_like(next_image_unreferenced)

                # get the text boxes
                next_text_boxes = next_data['text_bbox'].iloc[i].split(";")

                # Convert string representation to list of tuples
                next_coordinate_pairs = [tuple(map(lambda x: int(float(x)), coord[1:-1].split(',')))
                                         for coord in next_text_boxes]

                # Set the regions in the array to 0
                for top_left_x, top_left_y, bottom_right_x, bottom_right_y in next_coordinate_pairs:
                    next_mask_unreferenced[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

                # remove the borders
                next_mask_unreferenced = rb.remove_borders(next_mask_unreferenced, next_ids[i])

                # get the angle of next image
                sql_string = f"SELECT azimuth FROM images WHERE image_id='{next_ids[i]}'"
                next_azimuth_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)
                next_angle = next_azimuth_data['azimuth'].iloc[0]

                # rotate
                next_mask_unreferenced = ri.rotate_image(next_mask_unreferenced, next_angle, start_angle=0)

                # transform the mask
                # Open a new in-memory file for writing.
                with MemoryFile() as memfile:
                    with memfile.open(driver='GTiff',
                                      height=next_image.shape[0], width=next_image.shape[1],  # noqa
                                      count=1, dtype=str(next_mask_unreferenced.dtype),
                                      crs='EPSG:3031', transform=next_transform) as dataset:  # noqa
                        # Write the mask to the in-memory file.
                        dataset.write(next_mask_unreferenced, 1)

                    # Open the in-memory file for reading.
                    with memfile.open() as dataset:
                        # Now, dataset is a rasterio dataset that you can use as you would
                        # use a dataset opened from a file on disk.

                        next_mask = dataset.read()[0]
            else:
                next_mask = np.ones((2, 2))

            next_masks.append(next_mask)

    else:

        # create "dummy" masks
        mask = np.ones_like(image)

        for prev_footprint in prev_footprints_exact:
            if prev_footprint is None:
                prev_mask = np.ones((2, 2))
            else:
                prev_mask = np.ones_like(prev_image)
            prev_masks.append(prev_mask)

        for next_footprint in next_footprints_exact:
            if next_footprint is None:
                next_mask = np.ones((2, 2))
            else:
                next_mask = np.ones_like(next_image)
            next_masks.append(next_mask)

    if debug_show_images:

        images_to_show = [image] + prev_images + next_images
        masks_to_show = [mask] + prev_masks + next_masks
        titles = [image_id] + prev_ids + next_ids

        if mask_images:
            di.display_images(images_to_show + masks_to_show, x_titles=titles)
        else:
            di.display_images(images_to_show, titles=titles)

    # this numpy array and the list will later contain all tie points and their confs
    all_tps = np.empty([0, 4])
    all_conf = np.empty([0])

    # look for tie points between image and prev image
    for i, prev_footprint in enumerate(prev_footprints_exact):

        # check if we have a footprint
        if prev_footprint is None:
            continue

        # find tps between image and prev image
        prev_tps, prev_conf = ftp.find_tie_points(image, prev_images[i],
                                                  mask_1=mask, mask_2=prev_masks[i],
                                                  min_threshold=min_threshold,
                                                  additional_matching=True,
                                                  extra_matching=True,
                                                  catch=True, verbose=False, pbar=pbar)

        if prev_tps is not None:

            p.print_v(f"{prev_tps.shape[0]} tie-points found between {image_id} and {prev_ids[i]}",
                      verbose=verbose, pbar=pbar)

            if debug_show_tie_points_unfiltered and prev_tps.shape[0] > 0:
                dt.display_tiepoints([image, prev_images[i]], points=prev_tps, confidences=prev_conf,
                                     titles=[image_id, prev_ids[i]],
                                     title=f"{prev_tps.shape[0]} unfiltered tps found", verbose=False)

            if filter_tps and prev_tps.shape[0] > 0:
                prev_tps, prev_conf = fto.filter_tp_outliers(prev_tps, prev_conf,
                                                             use_ransac=filter_use_ransac,
                                                             ransac_threshold=filter_ransac_val,
                                                             use_angle=filter_use_angle,
                                                             angle_threshold=filter_angle,
                                                             verbose=verbose, pbar=pbar)

                if debug_show_tie_points_filtered and prev_tps.shape[0] > 0:
                    dt.display_tiepoints([image, prev_images[i]], points=prev_tps, confidences=prev_conf,
                                         titles=[image_id, prev_ids[i]],
                                         title=f"{prev_tps.shape[0]} filtered tps found", verbose=False)

            # apply transformation to the tie-points of the prev image, so that these tps are absolute
            prev_tps[:, 0] = prev_tps[:, 0] * prev_transforms[i][0]
            prev_tps[:, 1] = prev_tps[:, 1] * prev_transforms[i][4]
            prev_tps[:, 0] = prev_tps[:, 0] + prev_transforms[i][2]
            prev_tps[:, 1] = prev_tps[:, 1] + prev_transforms[i][5]

            # save prev tps to all tps
            all_tps = np.concatenate((all_tps, prev_tps), axis=0)

            # save prev conf to all conf
            all_conf = np.concatenate((all_conf, prev_conf))  # noqa

    # look for tie points between image and next image
    for i, next_footprint in enumerate(next_footprints_exact):

        # check if we have a footprint
        if next_footprint is None:
            continue

        # find tps between image and next image
        next_tps, next_conf = ftp.find_tie_points(image, next_images[i],
                                                  mask_1=mask, mask_2=next_masks[i],
                                                  min_threshold=min_threshold,
                                                  additional_matching=True,
                                                  extra_matching=True,
                                                  catch=True, verbose=False, pbar=pbar)

        if next_tps is not None:

            p.print_v(f"{next_tps.shape[0]} tie-points found between {image_id} and {next_ids[i]}",
                      verbose=verbose, pbar=pbar)

            if debug_show_tie_points_unfiltered and next_tps.shape[0] > 0:
                dt.display_tiepoints([image, next_images[i]], points=next_tps, confidences=next_conf,
                                     titles=[image_id, next_ids[i]],
                                     title=f"{next_tps.shape[0]} tps found", verbose=False)

            if filter_tps and next_tps.shape[0] > 0:
                next_tps, next_conf = fto.filter_tp_outliers(next_tps, next_conf,
                                                             use_ransac=filter_use_ransac,
                                                             ransac_threshold=filter_ransac_val,
                                                             use_angle=filter_use_angle,
                                                             angle_threshold=filter_angle,
                                                             verbose=verbose, pbar=pbar)

                if debug_show_tie_points_filtered and next_tps.shape[0] > 0:
                    dt.display_tiepoints([image, next_images[i]], points=next_tps, confidences=next_conf,
                                         titles=[image_id, next_ids[i]],
                                         title=f"{next_tps.shape[0]} filtered tps found", verbose=False)

            # apply transformation to the tie-points of the next image, so that these tps are absolute
            next_tps[:, 0] = next_tps[:, 0] * next_transforms[i][0]
            next_tps[:, 1] = next_tps[:, 1] * next_transforms[i][4]
            next_tps[:, 0] = next_tps[:, 0] + next_transforms[i][2]
            next_tps[:, 1] = next_tps[:, 1] + next_transforms[i][5]

            # save next tps to all tps
            all_tps = np.concatenate((all_tps, next_tps), axis=0)

            # save next conf to all conf
            all_conf = np.concatenate((all_conf, next_conf))  # noqa

    # filter all tps again
    if filter_tps and all_tps.shape[0] > 0:
        all_tps, all_conf = fto.filter_tp_outliers(all_tps, all_conf,
                                                   use_ransac=filter_use_ransac, ransac_threshold=filter_ransac_val,
                                                   use_angle=filter_use_angle, angle_threshold=filter_angle,
                                                   verbose=verbose, pbar=pbar)

    # check if we have enough tie-points
    if all_tps.shape[0] < min_tps_final:
        p.print_v(f"{image_id} has too few ({all_tps.shape[0]}) tie-points",
                  color="red", verbose=verbose, pbar=pbar)
        status = f"too_few_tps:{all_tps.shape[0]}"

    if debug_show_tie_points_final:
        print(image.shape, all_tps)
        di.display_images([image], points=all_tps[:, 2:4], point_size=25,
                          title=f"{all_tps.shape[0]} final tie-points found")

    if status.startswith("too_few_tps") is False:

        # get the transform
        transform, residuals = ag.apply_gcps(path_tiff, image, all_tps,
                                             transform_method, transform_order,
                                             conf=all_conf,
                                             return_error=True, save_image=debug_save,
                                             catch=catch)

        # check if the transform worked
        if transform is None:
            p.print_v(f"Something went wrong applying gcps to {image_id}", color="red",
                      verbose=verbose, pbar=pbar)
            return_tuple = (None, None, "failed:apply_transform", None, None)
            return return_tuple

        # get the exact footprint
        footprint = citf.convert_image_to_footprint(image, image_id, transform,
                                                    catch=catch, verbose=False, pbar=pbar)

        status = "georeferenced"
    else:
        footprint = None
        transform = None
        residuals = None

    # should we save the tps?
    if save_tie_points and all_tps.shape[0] > 0 and debug_save:
        # create a pandas dataframe with tie-point information
        # Create a DataFrame from the NumPy array
        df = pd.DataFrame({
            'x': all_tps[:, 0], 'y': all_tps[:, 1],
            'x_pic': all_tps[:, 2], 'y_pic': all_tps[:, 3],
            'conf': all_conf,
        })

        # convert to geopandas
        geometry = gpd.points_from_xy(x=df['x'], y=df['y'])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:3031')

        # save to file
        gdf.to_file(path_tie_points_shape, driver='ESRI Shapefile')

    return footprint, transform, status, all_tps.shape[0], residuals


if __name__ == "__main__":

    debug_save = False

    debug_show_images = False
    debug_show_tie_points_filtered = False
    debug_show_tie_points_final = False

    _path_fld = ""
    _path_georef_tiffs = "/data_1/ATM/data_1/playground/georef4/tiffs/sat"

    image_ids = ["CA183532V0084"]

    for _image_id in image_ids:
        _footprint, _transform, _status = georef_img(_image_id, _path_fld, _path_georef_tiffs,
                                                     catch=False, verbose=True)
