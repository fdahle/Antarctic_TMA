import copy

import numpy as np
import pandas as pd

from tqdm import tqdm

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.rotate_image as ri

import image_georeferencing.sub.adjust_image_resolution as air
import image_georeferencing.sub.buffer_footprint as bf
import image_georeferencing.sub.calc_camera_position as ccp
import image_georeferencing.sub.calc_footprint_approx as cfa
import image_georeferencing.calc_footprint_real as cfr
import image_georeferencing.old.create_similarity_heatmap as csh
import image_georeferencing.old.georeference_single_image as gsi
import image_georeferencing.old.georeference_image_with_satellite as giws
import image_georeferencing.old.georeference_image_with_ortho as giwo
import image_georeferencing.old.georeference_image_with_image as giwi
import image_georeferencing.sub.load_satellite_data as lsd
import image_georeferencing.sub.merge_footprints as mf

_path_ortho = ""
img_ids = ["CA213732V0032", "CA213732V0033", "CA213732V0034"]
img_ids = ["CA214732V0030", "CA214732V0031", "CA214732V0032"]


def georeference_all_images(image_ids, path_orthophoto=None,
                            overwrite=False, catch=True, verbose=False):

    p.print_v(f"Start: georeference_all_images ({len(image_ids)} images)")

    # convert the list of image_ids to a string
    str_image_ids = "('" + "', '".join(image_ids) + "')"

    # get information from images
    sql_string = f"SELECT * FROM images WHERE image_id IN {str_image_ids}"
    data_images = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)

    # get information we need from image_extracted
    sql_string = f"SELECT * FROM images_extracted WHERE image_id IN {str_image_ids}"
    data_extracted = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose)

    # merge the information
    data = pd.merge(data_images, data_extracted, on='image_id')

    # here we save all the unique footprints and angles of every image
    footprints = []
    heights = []
    angles = []

    # first iterate all images to get the approx_footprint and angles
    for image_id in (pbar := tqdm(image_ids)):

        # get the row from the dataframe
        row = data.loc[data['image_id'] == image_id]

        if row.empty:
            p.print_v(f"Warning: Information missing in dataframe for '{image_id}'", verbose, pbar=pbar)
            if catch:
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

        # TEMP
        focal_length = 153.460

        # if we don't have a focal length for this image, we must skip this image
        if focal_length is None:
            p.print_v(f"Warning: Focal length missing for '{image_id}'",
                      verbose, pbar=pbar)
            if catch:
                continue
            else:
                raise ValueError(f"Warning: Focal length missing for '{image_id}'")

        # if we don't have a height for this image, we must skip this image
        if height is None:
            p.print_v(f"Warning: Height missing for '{image_id}'",
                      verbose, pbar=pbar)
            if catch:
                continue
            else:
                raise ValueError(f"Warning: Height missing for '{image_id}'")
        else:
            heights.append(height)

        # if we don't have an angle for this image, we must skip this image
        if angle is None:
            p.print_v(f"Warning: Angle missing for '{image_id}'",
                      verbose, pbar=pbar)
            if catch:
                continue
            else:
                raise ValueError(f"Warning: Angle missing for '{image_id}'")
        else:
            angles.append(angle)

        # if the approx_footprint is not existing yet, we need to calculate it
        if footprint_approx is None:
            # get the shapefile position information
            x = row["x_coords"]
            y = row["y_coords"]

            # calculate an approximate approx_footprint based on the information we have from the shapefile
            footprint_approx = cfa.calc_footprint_approx(x, y, angle, view_direction,
                                                         height, focal_length,
                                                         catch=catch, verbose=verbose, pbar=pbar)

            # save this approx_footprint in the database
            #success = utif.update_table_images_footprints(image_id, "footprint_approx", footprint_approx,
            #                                              overwrite=overwrite, catch=catch, verbose=verbose, pbar=pbar)

            #if success is None:
            #    lst_failed_ids.append([image_id, "save_footprint_approx"])

        footprints.append(footprint_approx)

    # check if the angles are similar
    if np.std(angles) > 5:
        if verbose:
            print("Warning: High standard deviation for angles")

    # get the average angle (used to rotate the ortho-photo)
    angle = np.average(angles)

    # merge footprints and buffer
    footprint_merged = mf.merge_footprints(footprints, catch=catch, verbose=verbose)
    footprint_buffered = bf.buffer_footprint(footprint_merged, catch=catch, verbose=verbose)

    # get bounds from approx_footprint
    sat_bounds = footprint_buffered.bounds

    # get the satellite image of the approx_footprint
    sat_image, sat_transform = lsd.load_satellite_data(sat_bounds,
                                                       return_transform=True,
                                                       catch=catch, verbose=False)

    print(sat_image.shape)

    """
    sat_orig = copy.deepcopy(sat_image)
    import cv2
    # Convert to HSV color space
    sat_image = np.transpose(sat_image, (1, 2, 0))
    hsv = cv2.cvtColor(sat_image, cv2.COLOR_RGB2HSV)

    # Extract luminance component (value channel)
    value = hsv[:, :, 2]

    # Apply threshold to create binary mask
    mask = np.logical_and(value >= 75, value <= 110)
    # Convert the mask to a uint8 array
    mask = mask.astype(np.uint8) * 255

    # Apply inpainting to fill in shadowed regions
    sat_image = cv2.inpaint(sat_image, mask, 5, cv2.INPAINT_TELEA)

    di.display_images([sat_orig, sat_image])
    exit()
    """

    # now try to find the aerial image in this satellite image
    for i, image_id in (pbar := tqdm(enumerate(image_ids))):

        # load image and rotate
        img = liff.load_image_from_file(image_id, catch=catch, verbose=verbose, pbar=pbar)

        # adjust the image so that pixel size is similar to satellite image
        img_adjusted = air.adjust_image_resolution(sat_image, sat_bounds, img, footprints[i].bounds)

        # get the edge dimensions
        _, edge_dims = rb.remove_borders(img, image_id=image_id, return_edge_dims=True,
                                         catch=catch, verbose=verbose)

        # adapt the edge dimension to account for smaller image
        change_y = img.shape[0] / img_adjusted.shape[0]
        change_x = img.shape[1] / img_adjusted.shape[1]
        if change_y > 0 or change_x > 0:
            change_y = 1 / change_y
            change_x = 1 / change_x
        edge_dims[0] = int(edge_dims[0] * change_x)
        edge_dims[1] = int(edge_dims[1] * change_x)
        edge_dims[2] = int(edge_dims[2] * change_y)
        edge_dims[3] = int(edge_dims[3] * change_y)

        img_adjusted = img_adjusted[edge_dims[2]:edge_dims[3], edge_dims[0]:edge_dims[1]]

        from PIL import Image
        img_temp = copy.deepcopy(img_adjusted)
        image_temp = Image.fromarray(img_temp)
        image_temp = image_temp.rotate(360 - angles[i] + 90, expand=True)
        img_adjusted_rotated = np.asarray(image_temp)  # noqa

        import image_tie_points.find_tie_points as ftp
        points_all, conf_all = ftp.find_tie_points(sat_image, img_adjusted_rotated,
                                                     mask_1=None, mask_2=None,
                                                     min_threshold=0.2,
                                                     extra_mask_padding=10,
                                                     additional_matching=True,
                                                     extra_matching=True,
                                                     keep_resized_points=True,
                                                     keep_additional_points=True,
                                                     catch=catch, verbose=verbose)
        import display.display_tiepoints as dt
        dt.display_tiepoints([sat_image, img_adjusted_rotated], points_all, conf_all)

    exit()

    print(sat_image.shape)

    print(path_orthophoto)

    # load the ortho-photo
    ortho = liff.load_image_from_file(path_orthophoto, catch=catch, verbose=verbose)

    # rotate the ortho-photo
    ortho, ortho_mask = ri.rotate_image(ortho, angle, return_mask=True,
                                        catch=catch, verbose=verbose)

    # TODO: HEATMAP FOR ORTHO TO REDUCE SATELLITE IMAGE
    similarity_heatmap = csh.create_similarity_heatmap(sat_image, ortho,
                                                       catch=catch, verbose=verbose)

    # try to georeference the ortho-photo based on the satellite image
    # def georeference_single_image(georef_img, geo_ref_bounds, geo_ref_transform,
    #                              nonref_img, nonref_id, output_fld, output_fld_gcps=None,
    #                              transform_method="gdal", gdal_order="auto",
    #                              georef_mask=None, nonref_mask=None):

    # geo-reference ortho-photo
    gsi.georeference_single_image(sat_image, sat_bounds, sat_transform,
                                  ortho, "",
                                  nonref_mask=ortho_mask,
                                  catch=catch, verbose=verbose)

    # now we want to geo-reference the single images
    for image_id in (pbar := tqdm(image_ids)):

        # get the image
        img = liff.load_image_from_file(image_id,
                                        catch=catch, verbose=verbose, pbar=pbar)

        # cut off the image borders
        img = rb.remove_borders(img, image_id=image_id,
                                catch=catch, verbose=verbose, pbar=pbar)

        # rotate the image
        img = ri.rotate_image(img, angle, return_mask=True,
                              catch=catch, verbose=verbose, pbar=pbar)

        # georeference image with satellite
        georeferenced_image_sat = giws.georeference_image_with_satellite(img)

        # georeference image with ortho
        georeferenced_image_ort = giwo.georeference_image_with_ortho(img)

        # georeference image with another image
        georeferenced_image_img = giwi.georeference_image_with_image(img)

        real_footprint = cfr.calculate_footprint_real()

        real_camera_position = ccp.calculate_camera_position()


if __name__ == "__main__":
    georeference_all_images(img_ids, catch=False, verbose=True)