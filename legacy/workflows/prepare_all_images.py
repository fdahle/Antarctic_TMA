import json
import os

from tqdm import tqdm

import image_preparation.correct_image_orientation_sidebar as cios
import image_preparation.create_image_entry_in_database as cieid
import image_preparation.extract_fid_marks as efm1
import image_preparation.extract_subsets as es1
import image_preparation.download_image_from_usgs as difu
import image_preparation.remove_usgs_logo as rul
import image_preparation.update_table_images as uti
import image_preparation.update_table_images_fid_marks as utifm

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.load_shape_data as lsd
import base.print_v as p
import base.update_failed_ids as ufi

debug_load_shape_data = True
debug_download = False
debug_create_entry = False
debug_remove_logo = False
debug_correct_image_orientation = False
debug_update_basic_table = False
debug_extract_subsets = True
debug_extract_fid_marks = True


def prepare_all_images(image_ids, path_shape_file=None,
                       overwrite=False, catch=True, verbose=False):
    """
    prepare_all_images(image_ids, overwrite, catch, verbose):
    This function calls all functions that are necessary to prepare the images for further
    usage in this project. See the specific functions to see what is happening
    Args:
        image_ids (List): A list of image_ids
        path_shape_file (String): path to the shapefile, in which information about the images is located
        overwrite (Boolean): If true, all values and data will be overwritten with new values,
            if false, existing values and data will not be changed
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of
            the function
    Returns:
         -
    """

    p.print_v(f"Start: prepare_all_images ({len(image_ids)} images)", verbose=verbose)

    # load the json to get default values
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_folder = project_folder + "/image_preparation"
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # get the path to the shape file
    if path_shape_file is None:
        path_shape_file = json_data["path_file_photocenter_shapefile"]

    # get the shape data for the image
    if debug_load_shape_data:
        shape_data = lsd.load_shape_data(path_shape_file,
                                         catch=catch, verbose=verbose)

        # we necessarily need the shape data
        if shape_data is None:
            p.print_v(f"The shape data from '{path_shape_file}' could not be loaded. "
                      f"Please check your link to the data.", verbose=True, color="red")
            exit()
    else:
        shape_data = None

    # initialize the failed image_id manager
    fail_manager = ufi.FailManager("prepare_all_images")

    # iterate all image_ids
    for image_id in (pbar := tqdm(image_ids)):

        p.print_v(f"Runtime: prepare_all_images: {image_id}", verbose=verbose, pbar=pbar)

        # get the image path
        img_path = json_data["path_folder_downloaded"] + image_id + ".tif"

        if debug_download:
            # download the image from usgs
            success = difu.download_image_from_usgs(image_id, overwrite=overwrite,
                                                    catch=catch, verbose=verbose, pbar=pbar)

            # check if something went wrong with downloading
            if success is False:
                fail_manager.update_failed_id(image_id, "download_image_from_usgs")
                continue

        if debug_create_entry:
            # now that we have downloaded the image, create an entry for the image in the database
            # (for images, images_fid_points & images_extracted)
            success = cieid.create_image_entry_in_database(image_id, overwrite=overwrite,
                                                           catch=catch, verbose=verbose, pbar=pbar)

            # check if it was possible to create an entry
            if success is False:
                fail_manager.update_failed_id(image_id, "create_image_entry_in_database")
                continue

            # update the 'images' table
            success = uti.update_table_images(image_id, img_path, shape_data, overwrite=overwrite,
                                              catch=catch, verbose=verbose, pbar=pbar)

            # check if it was possible to update table images
            if success is False:
                fail_manager.update_failed_id(image_id, "update_table_images")
                continue

        # load the image
        image = liff.load_image_from_file(image_id,
                                          catch=catch, verbose=verbose,
                                          pbar=pbar)

        # check if something went wrong with loading the image
        if image is None:
            fail_manager.update_failed_id(image_id, "load_image_from_file")
            continue

        # remove the usgs logo
        if debug_remove_logo:
            image = rul.remove_usgs_logo(image, image_id=image_id, overwrite=overwrite,
                                         catch=catch, verbose=verbose, pbar=pbar)

            # check if it was possible to remove the logo
            if image is None:
                fail_manager.update_failed_id(image_id, "remove_usgs_logo")
                continue

        if debug_correct_image_orientation:
            # correct the image orientation, so that the sideline is on the left-side
            image, success = cios.correct_image_orientation_sidebar(image, image_id=image_id, overwrite=overwrite,
                                                                    catch=catch, verbose=verbose, pbar=pbar)

            # check if it was possible to correct the image orientation
            if success is False:
                fail_manager.update_failed_id(image_id, "correct_orientation_sidebar")
                continue

        if debug_update_basic_table:
            # prepare the update data for the basic image properties
            update_data = {
                "image_width": image.shape[1],
                "image_height": image.shape[0]
            }

            # update the table image fid_points with the basic image properties
            success = utifm.update_table_images_fid_marks(image_id, "basic", update_data, overwrite=overwrite,  # noqa
                                                          catch=catch, verbose=verbose, pbar=pbar)

            # check if it was possible to update table image_fid_points
            if success is False:
                fail_manager.update_failed_id(image_id, "update_table_images_fid_marks:basic")
                continue

        if debug_extract_subsets:

            # we want to try to find subsets with both non- and binarized images
            for binarize in [False, True]:

                # extract subsets
                subset_data = es1.extract_subsets(image, image_id, binarize_subset=binarize,
                                                  overwrite=overwrite,
                                                  catch=catch, verbose=verbose, pbar=pbar)

                # check if we could extract subsets
                if subset_data is None:
                    fail_manager.update_failed_id(image_id, "extract_subsets")
                    continue

                success = utifm.update_table_images_fid_marks(image_id, "subsets", subset_data,
                                                              overwrite=overwrite,
                                                              catch=catch, verbose=verbose, pbar=pbar)

                # check if it was possible to update table image_fid_points
                if success is False:
                    fail_manager.update_failed_id(image_id, "update_table_images_fid_marks:subsets")
                    continue

        else:

            # get subset data
            sql_string = f"SELECT * FROM images_fid_points WHERE image_id='{image_id}'"
            data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

            data = data.iloc[0]

            # convert to dict
            subset_data = {}
            for key in ["n", "e", "s", "w"]:
                min_x = data[f"subset_{key}_x"]
                min_y = data[f"subset_{key}_y"]
                if min_x is None or min_y is None:
                    subset_data[key] = None
                else:
                    max_x = min_x + json_data["subset_width_px"]
                    max_y = min_y + json_data["subset_height_px"]
                    subset_data[key] = [min_x, max_x, min_y, max_y]

        if debug_extract_fid_marks:
            # extract fiducial marks
            fid_data = efm1.extract_fid_marks(image, image_id, subset_data,
                                              catch=catch, verbose=verbose, pbar=pbar)

            # check if we could extract fid marks
            if fid_data is None:
                fail_manager.update_failed_id(image_id, "extract_fid_marks")
                continue

            success = utifm.update_table_images_fid_marks(image_id, "fid_marks", fid_data,
                                                          overwrite=overwrite,
                                                          catch=catch, verbose=verbose, pbar=pbar)

            # check if it was possible to update table image_fid_marks
            if success is False:
                fail_manager.update_failed_id(image_id, "update_table_images_fid_marks:fid_marks")
                continue

        # if everything worked remove from failed ids
        fail_manager.remove_failed_id(image_id)

    # save the failed ids
    fail_manager.save_csv()

    p.print_v(f"Finished: prepare_all_images ({len(image_ids)} images)", verbose=verbose)


if __name__ == "__main__":

    """
    file_path = "/data_1/ATM/data_1/playground/img_ids_peninsula.txt"
    with open(file_path, "r") as file:
        # Read the entire contents of the file
        text = file.read()
        text = text.strip("\n")

    img_ids = text.split(";")
    """

    img_ids = ['CA180031L0019', 'CA512333R0055', 'CA512333R0126', 'CA512733R0096', 'CA184731L0001', 'CA512731L0096',
               'CA180131L0095', 'CA512331L0101', 'CA180031L0054', 'CA204133R0108', 'CA512733R0110', 'CA204133R0041',
               'CA512433R0035']

    import random
    random.shuffle(img_ids)
    prepare_all_images(img_ids, overwrite=False, catch=True, verbose=True)
