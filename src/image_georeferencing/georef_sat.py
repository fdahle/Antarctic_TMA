import copy
import geopandas as gpd
import numpy as np
import pandas as pd

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.print_v as p
import base.remove_borders as rb
import base.resize_image as rei
import base.rotate_image as ri

import image_georeferencing.sub.adjust_image_resolution as air
import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.convert_image_to_footprint as citf
import image_georeferencing.sub.enhance_image as eh
import image_georeferencing.sub.filter_tp_outliers as fto
import image_georeferencing.sub.find_footprint_direction as ffd
import image_georeferencing.sub.load_satellite_data as lsd

import image_tie_points.find_tie_points as ftp

import display.display_images as di
import display.display_tiepoints as dt

# debug params to show the images/tie-points
debug_show_initial_images = False
debug_show_initial_tie_points = False
debug_show_located_tie_points = False
debug_show_tweaked_tie_points = False
debug_show_unfiltered_tie_points = False
debug_show_filtered_tie_points = False
debug_show_final_tie_points = False

# some other debug params
debug_save = True


def georef_sat(image_id, path_fld,
               approx_footprint, azimuth, month=0,
               existing_tiepoints_shape=None,
               min_tps_running=5, min_tps_final=25,
               mask_images=True,
               enhance_image=True, enhancement_min=0.02, enhancement_max=0.98,
               locate_image=True, location_max_order=3, location_overlap=1 / 3,
               tweak_image=True, tweaking_max_iterations=10,
               filter_tps=True, filter_use_ransac=True, filter_ransac_val=10,
               filter_use_angle=True, filter_angle=10,
               use_small_image_size=False,
               transform_method="rasterio", transform_order=3,
               save_tie_points=True,
               catch=True, verbose=False, pbar=None):
    """
    georef_sat():
    geo-reference an image based on a satellite image

    Args:
        image_id (String): The id of the image we want to geo-reference
        path_fld (String): Where do we want to store the temporary tif, png & shapefile
        approx_footprint (Shapely-polygon): A polygon describing the approximate position of the image
        azimuth (int): The approximate rotation of the image
        month (int, 0): Should we load satellite images from a certain month (if 0 load composite over whole year)
        existing_tiepoints_shape (geopandas, None): If we already estimated tie-points, we can use data
            from an existing shapefile
        min_tps_running (int, 5): If we are below this number during run-time, we skip this image
        min_tps_final (int, 25): We need at least that many tps to geo-reference
        mask_images (Boolean, true): Should a mask be applied on the image when looking for matches
        enhance_image (Boolean, true): should the image be enhanced before matching tie-points
        enhancement_min (float, 0.05): value for enhancement function
        enhancement_max (float, 0.95): value for enhancement function
        locate_image (Boolean, true): should the image be located in the vicinity
        location_max_order (int, 3): # how many layers of search should be used for locating the image
        location_overlap (float, 1/3): # how much should the tiles of location search overlap
        tweak_image (Boolean, true): If true, the sat bounds are tweaked to find more tps
        tweaking_max_iterations (int, 10): maximum number an image can be tweaked
        filter_tps (Boolean, true): If true, tps are checked for outliers
        filter_use_ransac (Boolean, true): If true, ransac is used for outlier filtering
        filter_ransac_val (int, 10): Maximum allowed re-projection error for ransac
        filter_use_angle (Boolean, true): If true, outliers are filtered with angles
        filter_angle (int, 10): Maximum allowed deviation for angle filtering
        use_small_image_size (Boolean, false): If true, images keep the adjusted resolution
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
    path_tie_points_img = path_fld + "/" + image_id + "_points.png"
    path_tie_points_shape = path_fld + "/" + image_id + "_points.shp"

    p.print_v(f"Load {image_id} from file", verbose=verbose, pbar=pbar)

    # load the image
    image_with_borders = liff.load_image_from_file(image_id, catch=True, verbose=False)

    # if something went wrong loading the image
    if image_with_borders is None:
        if catch:
            p.print_v(f"Error loading {image_id}", verbose=verbose, pbar=pbar, color="red")
            return_tuple = (None, None, "failed:load_image", None, None)
            return return_tuple
        else:
            raise ValueError(f"Error loading {image_id}")

    p.print_v(f"Remove borders for {image_id}", verbose=verbose, pbar=pbar)

    # get the border dimensions
    image_no_borders, border_dims = rb.remove_borders(image_with_borders, image_id=image_id,
                                                      return_edge_dims=True,
                                                      catch=True, verbose=False)

    # if something went wrong loading the borders
    if image_no_borders is None:
        if catch:
            p.print_v(f"Error removing borders from {image_id}", verbose=verbose, pbar=pbar, color="red")
            return_tuple = (None, None, "failed:remove_borders", None, None)
            return return_tuple
        else:
            raise ValueError(f"Error removing borders from {image_id}")

    p.print_v(f"Rotate {image_id}", verbose=verbose, pbar=pbar)

    # rotate the image
    image_no_borders_rotated = ri.rotate_image(image_no_borders, azimuth, start_angle=0,
                                               catch=True, verbose=False)
    # if we cannot rotate the image
    if image_no_borders_rotated is None:
        if catch:
            p.print_v(f"Error rotating {image_id}", verbose=verbose, pbar=pbar, color="red")
            return_tuple = (None, None, "failed:rotate_image", None, None)
            return return_tuple
        else:
            raise ValueError(f"Error rotating {image_id}")

    # we already have tie-points
    if existing_tiepoints_shape:

        # extract the data we need from the shapefile
        tps = existing_tiepoints_shape[['x_sat', 'y_sat', 'x_pic', 'y_pic']].to_numpy()
        tps_abs = existing_tiepoints_shape[['x', 'y', 'x_pic', 'y_pic']].to_numpy()
        conf = existing_tiepoints_shape[['conf']].tolist()
        image = image_no_borders_rotated
        # image_enhanced = image  # we need this just for display reasons

        change_x_unrotated = existing_tiepoints_shape.loc[0, ['x_fac_urot']]  # noqa
        change_y_unrotated = existing_tiepoints_shape.loc[0, ['y_fac_urot']]  # noqa
        change_x_rotated = existing_tiepoints_shape.loc[0, ['x_fac_rot']]  # noqa
        change_y_rotated = existing_tiepoints_shape.loc[0, ['y_fac_rot']]  # noqa

        sat_bounds = existing_tiepoints_shape.loc[0, ['sat_min_x', 'sat_min_y',
                                                      'sat_max_x', 'sat_max_y']].tolist()

        sat = lsd.load_satellite_data(sat_bounds, month=month, fallback_month=True, catch=True)

    # we need to calculate tie-points
    else:

        # create a mask for the image
        if mask_images:

            # create mask and remove borders
            mask = np.ones_like(image_with_borders)
            mask[:border_dims[2], :] = 0
            mask[border_dims[3]:, :] = 0
            mask[:, :border_dims[0]] = 0
            mask[:, border_dims[1]:] = 0

            # get data for text boxes
            # get the data of the image
            sql_string = f"SELECT text_bbox FROM images_extracted WHERE image_id='{image_id}'"
            data = ctd.get_data_from_db(sql_string, catch=True)
            text_data = data.iloc[0]['text_bbox']

            # get the text boxes from data
            text_boxes = text_data.split(";")

            # Convert string representation to list of tuples
            coordinate_pairs = [tuple(map(lambda x: int(float(x)), coord[1:-1].split(','))) for coord in text_boxes]

            # Set the regions in the array to 0
            for top_left_x, top_left_y, bottom_right_x, bottom_right_y in coordinate_pairs:
                mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0

        else:
            mask = np.ones_like(image_with_borders)

        # rotate the mask
        mask = ri.rotate_image(mask, azimuth, start_angle=0, catch=True, verbose=False)  # noqa

        # get the initial satellite bounds from the approx_footprint
        sat_bounds = list(approx_footprint.bounds)
        sat_bounds = [int(x) for x in sat_bounds]

        p.print_v(f"Load satellite image at {sat_bounds}", verbose=verbose, pbar=pbar)

        # load the initial approx satellite image
        sat, sat_transform, lst_sat_images = lsd.load_satellite_data(sat_bounds,
                                                                     month=month,
                                                                     fallback_month=True,
                                                                     return_transform=True,
                                                                     return_used_images=True,
                                                                     catch=True, verbose=False,
                                                                     pbar=pbar)

        # if the satellite is none means we don't have a satellite image for it
        if sat is None:
            p.print_v(f"Could not load satellite image for {image_id}", verbose=verbose,
                      pbar=pbar, color="red")
            return_tuple = (None, None, "failed:load_satellite_data", None, None)
            return return_tuple

        # adjust the image so that pixel size is similar to satellite image
        image_with_borders_adjusted = air.adjust_image_resolution(sat,
                                                                  sat_bounds,
                                                                  image_with_borders,
                                                                  approx_footprint.bounds)

        # get the difference between the images with borders (unrotated)
        change_y_unrotated = image_with_borders.shape[0] / image_with_borders_adjusted.shape[0]
        change_x_unrotated = image_with_borders.shape[1] / image_with_borders_adjusted.shape[1]

        # adapt the border dimension to account for smaller image
        border_dims_adjusted = copy.deepcopy(border_dims)
        if change_y_unrotated > 0 or change_x_unrotated > 0:
            change_y_unrotated = 1 / change_y_unrotated
            change_x_unrotated = 1 / change_x_unrotated
        border_dims_adjusted[0] = int(border_dims[0] * change_x_unrotated)
        border_dims_adjusted[1] = int(border_dims[1] * change_x_unrotated)
        border_dims_adjusted[2] = int(border_dims[2] * change_y_unrotated)
        border_dims_adjusted[3] = int(border_dims[3] * change_y_unrotated)

        # remove borders from image
        image_no_borders_adjusted = image_with_borders_adjusted[
                                    border_dims_adjusted[2]:border_dims_adjusted[3],
                                    border_dims_adjusted[0]:border_dims_adjusted[1]
                                    ]

        # rotate image
        image_no_borders_adjusted_rotated = ri.rotate_image(image_no_borders_adjusted, azimuth, start_angle=0)

        # get the difference between the images with no borders (rotated)
        change_y_rotated = image_no_borders_rotated.shape[0] / image_no_borders_adjusted_rotated.shape[0]
        change_x_rotated = image_no_borders_rotated.shape[1] / image_no_borders_adjusted_rotated.shape[1]
        if change_y_rotated > 0 or change_y_rotated > 0:
            change_y_rotated = 1 / change_y_rotated
            change_x_rotated = 1 / change_x_rotated

        # adapt the mask size
        mask = rei.resize_image(mask, image_no_borders_adjusted_rotated.shape)

        # should we enhance the image to find more tie-points?
        if enhance_image:
            image_enhanced = eh.enhance_image(image_no_borders_adjusted_rotated,
                                              scale=(enhancement_min, enhancement_max))
        # otherwise use the original
        else:
            image_enhanced = copy.deepcopy(image_no_borders_adjusted_rotated)

        # should we show the initial images
        if debug_show_initial_images:
            di.display_images([sat, image_enhanced], title=f"Initial approx_footprint for {image_id}")

        # find initial tie-points
        tps, conf = ftp.find_tie_points(sat, image_enhanced,
                                        min_threshold=0,
                                        filter_outliers=False,
                                        additional_matching=True, extra_matching=True,
                                        catch=True, verbose=False, pbar=None)

        # catch error in tie-point calculation
        if tps is None or conf is None:
            if catch:
                p.print_v(f"Could not get initial tie-points for {image_id}", verbose=verbose,
                          pbar=pbar, color="red")
                return_tuple = (None, None, "failed:get_initial_tie_points", None, None)
                return return_tuple
            else:
                raise ValueError(f"Could not get initial tie-points for {image_id}")

        # show the initial tie-points
        if tps.shape[0] > 0 and debug_show_initial_tie_points:
            dt.display_tiepoints([sat, image_enhanced], tps, conf, verbose=False,
                                 title=f"Tie-points for {image_id} ({tps.shape[0]})")

        # should we locate the image in the vicinity?
        if locate_image and tps.shape[0] < min_tps_final:

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
            best_nr_of_points = tps.shape[0]
            best_avg_conf = np.mean(np.asarray(conf)) if len(conf) > 0 else 0

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

                    # get the satellite image
                    sat_combi, sat_transform_combi, lst_sat_images = lsd.load_satellite_data(sat_bounds_combi,
                                                                                             return_transform=True,
                                                                                             return_used_images=True,
                                                                                             catch=True,
                                                                                             verbose=False,
                                                                                             pbar=None)

                    if sat_combi is None:
                        p.print_v(f"No satellite image could be found for combination {combination}",
                                  verbose=verbose, pbar=pbar, color="red")

                    else:

                        # get tie-points between satellite image and historical image
                        tps_combi, conf_combi = ftp.find_tie_points(sat_combi, image_enhanced,
                                                                    mask_2=mask,
                                                                    min_threshold=0,
                                                                    filter_outliers=False,
                                                                    additional_matching=True,
                                                                    extra_matching=True,
                                                                    catch=True, verbose=False, pbar=None)

                        # account for invalid tie-point-matching
                        if conf_combi is None:
                            conf_combi = []

                        nr_points_combi = 0 if tps_combi is None else tps_combi.shape[0]
                        avg_conf_combi = np.mean(np.asarray(conf_combi)) if len(conf_combi) > 0 else 0

                        p.print_v(f"{nr_points_combi} tie-points ({avg_conf_combi}) found at {combination}",
                                  verbose=verbose, pbar=pbar)

                        # check if we have the best combination
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

                        if best_nr_of_points > min_tps_final:
                            p.print_v(f"{best_nr_of_points} found in an combination -> "
                                      f"stop searching this combination!",
                                      verbose=verbose, pbar=pbar)
                            break

                if best_nr_of_points > min_tps_final:
                    break

            # extract best values
            tps = copy.deepcopy(best_tps)
            conf = copy.deepcopy(best_conf)
            sat = copy.deepcopy(best_sat)
            sat_transform = copy.deepcopy(best_sat_transform)
            sat_bounds = copy.deepcopy(best_sat_bounds)

            p.print_v(f"Best combination is {best_combination} with "  # noqa
                      f"{best_nr_of_points} tie-points ({best_avg_conf})",
                      verbose=verbose, pbar=pbar)  # noqa

            if best_tps.shape[0] > 0 and debug_show_located_tie_points:
                dt.display_tiepoints([sat, image_enhanced], best_tps, best_conf, verbose=False,
                                     title=f"Tie-points for {image_id} ({best_tps.shape[0]})")

        # check number of tps if continuing is worth it
        if tps.shape[0] < min_tps_running:
            p.print_v(f"Too few tie-points after locate_image ({tps.shape[0]}) to georeference {image_id}",
                      verbose=verbose, pbar=pbar, color="red")
            status = f"too_few_tps:{tps.shape[0]}"

        # should we tweak the images?
        if tweak_image and status.startswith("too_few_tps") is False:
            p.print_v(f"Start tweaking satellite image for {image_id}", verbose=verbose, pbar=pbar)

            # get the start values
            sat_tweaked = copy.deepcopy(sat)
            sat_bounds_tweaked = copy.deepcopy(sat_bounds)
            tps_tweaked = copy.deepcopy(tps)

            # init the best values
            sat_tweaked_best = copy.deepcopy(sat)
            sat_bounds_tweaked_best = copy.deepcopy(sat_bounds)
            sat_transform_tweaked_best = copy.deepcopy(sat_transform)
            tweaked_best_nr_points = copy.deepcopy(tps.shape[0])

            tps_tweaked_best = copy.deepcopy(tps)
            conf_tweaked_best = copy.deepcopy(conf)

            # how often did we tweaked the image
            counter = 0

            # tweak the image in a loop
            while counter < tweaking_max_iterations:

                # we only want to tweak the image a maximum number of times
                counter = counter + 1

                # find the direction in which we need to change the satellite image
                step_y, step_x = ffd.find_footprint_direction(sat_tweaked, tps_tweaked[:, 0:2])

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
                tps_tweaked, conf_tweaked = ftp.find_tie_points(sat_tweaked, image_enhanced,
                                                                min_threshold=0,
                                                                filter_outliers=False,
                                                                additional_matching=True, extra_matching=True,
                                                                catch=True, verbose=False, pbar=None)

                if tps_tweaked is None:
                    p.print_v("Failed: finding tweaked tps", verbose=verbose, pbar=pbar, color="red")
                    return_tuple = (None, None, "failed:finding_tweaked_tps", None, None)
                    return return_tuple

                if debug_show_tweaked_tie_points:
                    dt.display_tiepoints([sat_tweaked, image_enhanced], tps_tweaked, conf_tweaked, verbose=False,
                                         title=f"Tweak nr {counter} with step {step_x, step_y}: Tie-points "
                                               f"for {image_id} ({tps_tweaked.shape[0]})")

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
                    p.print_v(f"Points going down ({tps_tweaked.shape[0]} < {tweaked_best_nr_points})",
                              verbose=verbose, pbar=pbar)
                    # stop the loop
                    break

            # restore the best settings again
            tps = copy.deepcopy(tps_tweaked_best)  # noqa
            conf = copy.deepcopy(conf_tweaked_best)  # noqa
            sat = copy.deepcopy(sat_tweaked_best)
            sat_transform = copy.deepcopy(sat_transform_tweaked_best)
            sat_bounds = copy.deepcopy(sat_bounds_tweaked_best)  # noqa

            if tps.shape[0] < min_tps_running:
                p.print_v(f"Too few tie-points after tweaking ({tps.shape[0]}) to georeference {image_id}",
                          verbose=verbose, pbar=pbar, color="red")
                status = f"too_few_tps:{tps.shape[0]}"

            else:
                p.print_v(f"Tweaking for {image_id} finished with {tps.shape[0]} tie-points",
                          verbose=verbose, pbar=pbar)

        if tps.shape[0] > 0 and debug_show_unfiltered_tie_points and status.startswith("too_few_tps") is False:
            dt.display_tiepoints([sat, image_enhanced], tps, conf, verbose=False,
                                 title=f"Unfiltered tie-points for {image_id} ({tps.shape[0]})")

        # filter with ransac and angle
        if filter_tps and status.startswith("too_few_tps") is False:
            tps, conf = fto.filter_tp_outliers(tps, conf,
                                               use_ransac=filter_use_ransac, ransac_threshold=filter_ransac_val,
                                               use_angle=filter_use_angle, angle_threshold=filter_angle,
                                               verbose=verbose, pbar=pbar)

            if tps.shape[0] > 0 and debug_show_filtered_tie_points:
                dt.display_tiepoints([sat, image_enhanced], tps, conf, verbose=False,
                                     title=f"Filtered Tie-points for {image_id} ({tps.shape[0]})")

            # intermediate check of the image
            if tps.shape[0] < min_tps_running:
                p.print_v(f"Too few tie-points after filtering ({tps.shape[0]}) to georeference {image_id}",
                          verbose=verbose, pbar=pbar, color="red")
                status = f"too_few_tps:{tps.shape[0]}"

        if use_small_image_size is False and status.startswith("too_few_tps") is False:

            # get the image with original image size
            image = copy.deepcopy(image_no_borders_rotated)

            # adapt the tps to account for the original image-size
            tps[:, 2] = (tps[:, 2] * (1 / change_x_rotated)).astype(int)
            tps[:, 3] = (tps[:, 3] * (1 / change_y_rotated)).astype(int)
        else:
            image = image_no_borders_adjusted_rotated

    # final check if we have found enough tie-points
    if tps.shape[0] < min_tps_final:
        status = f"too_few_tps:{tps.shape[0]}"

    # copy the tie-points for absolute coords
    tps_abs = copy.deepcopy(tps)
    if tps_abs.shape[0] > 0:
        absolute_points = np.array([sat_transform * tuple(point) for point in tps_abs[:, 0:2]])
        tps_abs[:, 0:2] = absolute_points

    if status.startswith("too_few_tps") is False:

        filter_tps_with_conf = False
        if filter_tps_with_conf:
            min_conf_val = 0.9
            min_tps_final = 25

            conf_np = np.array(conf)  # Convert conf list to numpy array

            # Initial filtering
            threshold_tps_abs = tps_abs[conf_np > min_conf_val]
            threshold_tps = tps[conf_np > min_conf_val]
            threshold_conf = conf_np[conf_np > min_conf_val]

            while len(threshold_tps_abs) < min_tps_final and min_conf_val > 0:
                min_conf_val -= 0.05
                threshold_tps_abs = tps_abs[conf_np > min_conf_val]
                threshold_tps = tps[conf_np > min_conf_val]
                threshold_conf = conf_np[conf_np > min_conf_val]

            tps_abs = threshold_tps_abs
            tps = threshold_tps
            conf = threshold_conf.tolist()

        # project satellite points to absolute coords
        # tps_abs[:, 0] = tps_abs[:, 0] * sat_transform[0]
        # tps_abs[:, 1] = tps_abs[:, 1] * sat_transform[4]
        # tps_abs[:, 0] = tps_abs[:, 0] + sat_transform[2]
        # tps_abs[:, 1] = sat_transform[5] - sat.shape[0] * sat_transform[
        #    4] + tps_abs[:, 1]

        # get the transform
        transform, residuals = ag.apply_gcps(path_tiff, image, tps_abs,
                                             transform_method, transform_order,
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
    if save_tie_points and tps.shape[0] > 0 and debug_save:
        # create a pandas dataframe with tie-point information
        # Create a DataFrame from the NumPy array
        df = pd.DataFrame({
            'x': tps_abs[:, 0], 'y': tps_abs[:, 1],
            'x_sat': tps[:, 0], 'y_sat': tps[:, 1],
            'x_pic': tps[:, 2], 'y_pic': tps[:, 3],
            'conf': conf,
            'x_fac_urot': change_x_unrotated, 'y_fac_urot': change_y_unrotated,  # noqa
            'x_fac_rot': change_x_rotated, 'y_fac_rot': change_y_rotated,
            'sat_min_x': sat_bounds[0], 'sat_max_x': sat_bounds[2],
            'sat_min_y': sat_bounds[1], 'sat_max_y': sat_bounds[3]
        })

        # convert to geopandas
        geometry = gpd.points_from_xy(x=df['x'], y=df['y'])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='epsg:3031')

        # save to file
        gdf.to_file(path_tie_points_shape, driver='ESRI Shapefile')

        # save tie-points also as image
        dt.display_tiepoints([sat, image], points=tps, confidences=conf,
                             title=f"Tie points for {image_id} ({tps.shape[0]})",
                             save_path=path_tie_points_img)

    if debug_show_final_tie_points:
        if tps.shape[0] > 0:
            dt.display_tiepoints([sat, image], tps, conf, verbose=False,
                                 title=f"Final Tie-points for {image_id} ({tps.shape[0]})")
        else:
            print("No final tie-points available for display")

    return_tuple = (footprint, transform, status, residuals, tps.shape[0])
    return return_tuple


if __name__ == "__main__":

    debug_save = True
    debug_show_unfiltered_tie_points = False
    debug_show_final_tie_points = True

    # image_ids = []

    # path = "/data_1/ATM/data_1/playground/georef3/tiffs/sat"
    # import os
    # for file in os.listdir(path):
    #     image_ids.append(file[:-4])

    image_ids = ["CA184832V0117"]

    for _image_id in image_ids:
        _path_fld = "/home/fdahle/Desktop/temp_georef"

        _sql_string = f"SELECT azimuth FROM images WHERE image_id='{_image_id}'"
        _data = ctd.get_data_from_db(_sql_string, catch=False)

        _sql_string_extracted = f"SELECT ST_AsText(footprint_approx) AS footprint_approx " \
                                f"FROM images_extracted WHERE image_id='{_image_id}'"
        _data_extracted = ctd.get_data_from_db(_sql_string_extracted, catch=False)

        import shapely.wkt

        _approx_footprint = shapely.wkt.loads(_data_extracted.iloc[0]['footprint_approx'])
        _azimuth = _data.iloc[0]['azimuth']

        _footprint, _transform, _status, res = georef_sat(_image_id, _path_fld, _approx_footprint,
                                                          _azimuth,
                                                          verbose=True)

        if _footprint is not None:
            import image_georeferencing.sub.check_georef_image as cgi

            import os

            print(_path_fld + "/" + _image_id + ".tif", os.path.isfile(_path_fld + "/" + _image_id + ".tif"))

            image_is_valid, invalid_reason = cgi.check_georef_image(_path_fld + "/" + _image_id + ".tif",
                                                                    return_reason=True,
                                                                    catch=True, verbose=True)

            print(image_is_valid, invalid_reason)
