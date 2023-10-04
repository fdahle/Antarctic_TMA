import copy
import json
import os

import pandas as pd

import base.connect_to_db as ctd
import base.print_v as p

import display.display_images as di

debug_show_borders = False


def remove_borders(input_img, image_id=None,
                   cut_method="auto", edge=None, extra_edge=None,
                   return_edge_dims=False, db_type="PSQL",
                   catch=True, verbose=False, pbar=None):
    """
    remove_borders(input_img, image_id, cut_method, edge, extra_edge, return_edge_dims, db_type,
                   catch, verbose, pbar):
    This function is cutting off the edge from images based on different methods. The edge are the black part of the
    images from the TMA archive that do not contain any semantic information. Note that the original input images are
    not changed (deep copy is applied before). The edges can be removed with a default value (cut_method "default",
    value based in 'edge') or more progressive based on fid points (cut_method "database", image_id required and needs
    fid points in all four corners of the image)

    Args:
        input_img (np-array): The raw image from where the edges should be cut off
        image_id (String, None): The image image_id of the input_img. Required if edges should be cut off based on fid points.
        cut_method (String, "default"): specifies the cut method, can be ["default", "database", "auto"]
        edge (int, None): The edge used when cutting via 'default'.
        extra_edge (int, None): Something you want to remove something extra on top of the calculated border
        return_edge_dims (Boolean, 'False'): if yes also the edges (what is cut off how much) is returned
        db_type: From where do we get the coordinates of the fid_points. Only required if cut_method='database'
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar

    Returns:
        img (np-array): The raw image with removed edges. If something went wrong and catch=True, 'None' will returned
        bounds [list]: how much is removed from the images from each side: x_left, x_right, y_top, y_bottom
    """

    p.print_v(f"Start: remove_borders ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if edge is None:
        edge = json_data["remove_borders_edge"]

    if extra_edge is None:
        extra_edge = json_data["remove_borders_extra_edge"]

    # deep copy to not change the original
    img = copy.deepcopy(input_img)

    # check prerequisites
    if cut_method == "database":
        assert image_id is not None, "to get the borders from the database, an image image_id is required"

    # check if correct method was chosen
    cut_methods = ["auto", "default", "database"]
    if cut_method not in cut_methods:
        p.print_v("The specified method is incorrect. Following methods are allowed:", color="red")
        p.print_v(cut_methods)
        exit()

    if image_id is None:
        p.print_v(f"Cut off edge for image with following method: {cut_method}",
                  verbose=verbose, pbar=pbar)
    else:
        p.print_v(f"Cut off edge for {image_id} with following method: {cut_method}",
                  verbose=verbose, pbar=pbar)

    # cut for default
    if cut_method == "default":

        try:
            # add extra edge to the default one
            edge = edge + extra_edge

            # save how much was cropped
            fid_points = {"fid_mark_1_x": edge,
                          "fid_mark_1_y": img.shape[1] - edge,
                          "fid_mark_2_x": img.shape[0] - edge,
                          "fid_mark_2_y": edge,
                          "fid_mark_3_x": edge,
                          "fid_mark_3_y": edge,
                          "fid_mark_4_x": img.shape[1] - edge,
                          "fid_mark_4_y": img.shape[0] - edge
                          }
            fid_points = pd.DataFrame(fid_points, index=[0])

            # get the right value from the dataframe
            min_x = max(fid_points["fid_mark_3_x"].iloc[0], fid_points["fid_mark_1_x"].iloc[0])
            max_x = min(fid_points["fid_mark_2_x"].iloc[0], fid_points["fid_mark_4_x"].iloc[0])
            min_y = max(fid_points["fid_mark_3_y"].iloc[0], fid_points["fid_mark_2_y"].iloc[0])
            max_y = min(fid_points["fid_mark_1_y"].iloc[0], fid_points["fid_mark_4_y"].iloc[0])

            bounds = [min_x, max_x, min_y, max_y]

        except (Exception, ) as e:
            if catch:
                if return_edge_dims:
                    return None, None
                else:
                    return None
            else:
                raise e

        # crop the image
        if catch:
            try:
                img = img[min_y:max_y, min_x:max_x]
            except (Exception,):
                if return_edge_dims:
                    return None, None
                else:
                    return None
        else:
            img = img[edge:img.shape[0] - edge, edge:img.shape[1] - edge]

    elif cut_method == "database":

        # build sql string to select the borders
        sql_string = "SELECT " \
                     "fid_mark_1_x, fid_mark_1_y, " \
                     "fid_mark_2_x, fid_mark_2_y, " \
                     "fid_mark_3_x, fid_mark_3_y, " \
                     "fid_mark_4_x, fid_mark_4_y " \
                     "FROM images_fid_points " + \
                     "WHERE image_id='" + image_id + "'"

        # get table_data
        table_data = ctd.get_data_from_db(sql_string, db_type=db_type,
                                          catch=catch, verbose=verbose, pbar=pbar)

        if table_data is None:
            if catch:
                if return_edge_dims:
                    return None, None
                else:
                    return None
            else:
                raise ValueError(f"Data from table is invalid ({image_id})")

        # check if there is any none value in the data
        bool_none_in_data = False
        for key in table_data:
            if table_data[key][0] is None:
                bool_none_in_data = True
                break

        # catch possible errors
        if bool_none_in_data:
            if catch:
                if return_edge_dims:
                    return None, None
                else:
                    return None
            else:
                raise ValueError(f"No border fid-points available ({image_id})")

        # get left
        if table_data["fid_mark_1_x"][0] >= table_data["fid_mark_3_x"][0]:
            left = table_data["fid_mark_1_x"][0]
        else:
            left = table_data["fid_mark_3_x"][0]

        # get top
        if table_data["fid_mark_2_y"][0] >= table_data["fid_mark_3_y"][0]:
            top = table_data["fid_mark_2_y"][0]
        else:
            top = table_data["fid_mark_3_y"][0]

        # get right
        if table_data["fid_mark_2_x"][0] <= table_data["fid_mark_4_x"][0]:
            right = table_data["fid_mark_2_x"][0]
        else:
            right = table_data["fid_mark_4_x"][0]

        # get bottom
        if table_data["fid_mark_1_y"][0] <= table_data["fid_mark_4_y"][0]:
            bottom = table_data["fid_mark_1_y"][0]
        else:
            bottom = table_data["fid_mark_4_y"][0]

        x_left = int(left + extra_edge)
        x_right = int(right - extra_edge)
        y_top = int(top + extra_edge)
        y_bottom = int(bottom - extra_edge)

        bounds = [x_left, x_right, y_top, y_bottom]

        try:
            img = img[y_top:y_bottom, x_left:x_right]
        except (Exception,):
            if catch:
                if return_edge_dims:
                    return None, None
                else:
                    return None
            else:
                raise ValueError("The extracted corner values do not fit the image")

    elif cut_method == "auto":

        db_img, db_bounds = remove_borders(input_img, image_id=image_id, cut_method="database",
                                           extra_edge=extra_edge, db_type=db_type,
                                           return_edge_dims=True,
                                           catch=True, verbose=verbose, pbar=pbar)

        if db_img is None:

            # try out the default cut off
            img, bounds = remove_borders(input_img, image_id=image_id, cut_method="default",
                                         extra_edge=extra_edge, return_edge_dims=True,
                                         catch=catch, verbose=verbose, pbar=pbar)

        else:
            img = db_img
            bounds = db_bounds

    else:
        p.print_v("That should not happen", verbose, color="red", pbar=pbar)

        # to satisfy the IDE
        bounds = None

    if debug_show_borders:
        di.display_images([input_img, img], title=f"Removed border for {image_id}")

    # return all stuff
    if return_edge_dims:
        return img, bounds
    else:
        return img
