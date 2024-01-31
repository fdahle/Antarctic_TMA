import json
import os

from tqdm import tqdm

import image_estimations.correct_focal_lengths as cfl
import image_estimations.deduct_focal_length as dfl
import image_estimations.estimate_cam_id as eci
import image_estimations.estimate_fid_marks as efm
import image_estimations.estimate_focal_length as efl
import image_estimations.estimate_height as eh
import image_estimations.estimate_lens_cone as elc
import image_estimations.estimate_subsets as es
import image_estimations.update_table_images_estimated as utie

import base.connect_to_db as ctd
import base.print_v as p
import base.update_failed_ids as ufi

debug_estimate_subsets = False
debug_estimate_fid_marks = False
debug_estimate_cam_id = False
debug_estimate_lens_cone = True
debug_estimate_focal_length = False
debug_estimate_height = False

debug_correct_focal_length = False

debug_deduct_focal_length = True


# first estimate focal_length, cam_id, lens-cone
# then correct focal_length, cam_id

def estimate_for_all_images(image_ids, overwrite=False, catch=True, verbose=False):
    p.print_v(f"Start: estimate_for_all_images ({len(image_ids)} images)", verbose=verbose)

    # load the json to get default values
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    json_folder = project_folder + "/image_estimations"
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)  # noqa

    # initialize the failed image_id manager
    fail_manager = ufi.FailManager("estimate_for_all_images")

    p.print_v("Estimate values:")

    # iterate all image_ids: estimation
    for image_id in (pbar := tqdm(image_ids)):

        p.print_v(f"Runtime: estimate_for_all_images: {image_id}", verbose=verbose, pbar=pbar)

        if debug_estimate_subsets:
            subset_estimations = es.estimate_subsets(image_id,
                                                     catch=catch, verbose=verbose, pbar=pbar)

            if subset_estimations is None:
                fail_manager.update_failed_id(image_id, "estimate_subsets")
                continue

        if debug_estimate_fid_marks:

            fid_mark_estimations = efm.estimate_fid_marks(image_id,
                                                          catch=catch, verbose=verbose, pbar=pbar)

            if fid_mark_estimations is None:
                fail_manager.update_failed_id(image_id, "estimate_fid_mark")
                continue

        if debug_estimate_cam_id:

            # first check if we need to estimate
            sql_string = "SELECT cam_id, cam_id_estimated FROM images_extracted " \
                         f"WHERE image_id='{image_id}'"
            data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
            count = data[(data['cam_id_estimated'] == True) |  # noqa, this must be == for pandas
                         (data['cam_id'].isnull())].shape[0]

            # only estimate if we must
            if count == 1 or overwrite is True:
                cam_id_estimation = eci.estimate_cam_id(image_id,
                                                        catch=catch, verbose=verbose,
                                                        pbar=pbar)

                if cam_id_estimation is None:
                    fail_manager.update_failed_id(image_id, "estimate_cam_id")
                else:
                    success = utie.update_table_images_estimated(image_id,
                                                                 "cam_id", cam_id_estimation,
                                                                 catch=catch, verbose=verbose, pbar=pbar)

                    if success is False:
                        fail_manager.update_failed_id(image_id, "update_table_cam_id")

        if debug_estimate_focal_length:

            # first check if we need to estimate
            sql_string = "SELECT focal_length, focal_length_estimated FROM images_extracted " \
                         f"WHERE image_id='{image_id}'"
            data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
            count = data[(data['focal_length_estimated'] == True) |  # noqa
                         (data['focal_length'].isnull())].shape[0]

            # only estimate if we must
            if count == 1 or overwrite is True:
                focal_length_estimation = efl.estimate_focal_length(image_id,
                                                                    catch=catch, verbose=verbose,
                                                                    pbar=pbar)

                if focal_length_estimation is None:
                    fail_manager.update_failed_id(image_id, "estimate_focal_length")
                else:
                    success = utie.update_table_images_estimated(image_id,
                                                                 "focal_length", focal_length_estimation,
                                                                 catch=catch, verbose=verbose, pbar=pbar)

                    if success is False:
                        fail_manager.update_failed_id(image_id, "update_table_focal_length")

        if debug_estimate_lens_cone:

            # first check if we need to estimate
            sql_string = "SELECT lens_cone, lens_cone_estimated FROM images_extracted " \
                         f"WHERE image_id='{image_id}'"
            data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
            count = data[(data['lens_cone_estimated'] == True) |  # noqa, this must be == for pandas
                         (data['lens_cone'].isnull())].shape[0]

            # only estimate if we must
            if count == 1 or overwrite is True:
                lens_cone_estimation = elc.estimate_lens_cone(image_id,
                                                              catch=catch, verbose=verbose,
                                                              pbar=pbar)

                if lens_cone_estimation is None:
                    fail_manager.update_failed_id(image_id, "estimate_lens_cone")
                else:
                    success = utie.update_table_images_estimated(image_id,
                                                                 "lens_cone", lens_cone_estimation,
                                                                 catch=catch, verbose=verbose, pbar=pbar)

                    if success is False:
                        fail_manager.update_failed_id(image_id, "update_table_lens_cone")

        if debug_estimate_height:
            height_estimation = eh.estimate_height(image_id,
                                                   catch=catch, verbose=verbose, pbar=pbar)

            if height_estimation is None:
                fail_manager.update_failed_id(image_id, "estimate_height")

            print("TODO UPDATE HEIGHT")

    # first correct for flight path
    if debug_correct_focal_length:
        p.print_v("Correct focal lengths:")
        cfl.correct_focal_lengths("flight_path")

    p.print_v("\nDeduct values:")

    # iterate all image_ids: deduct
    for image_id in (pbar := tqdm(image_ids)):

        if debug_deduct_focal_length:
            sql_string = "SELECT focal_length, focal_length_estimated FROM images_extracted " \
                         f"WHERE image_id='{image_id}'"
            data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)
            count = data[(data['focal_length_estimated'] == True) |  # noqa
                         (data['focal_length'].isnull())].shape[0]

            # only estimate if we must
            if count == 1 or overwrite is True:
                focal_length_deducted = dfl.deduct_focal_length(image_id,
                                                                catch=catch, verbose=verbose,
                                                                pbar=pbar)

                if focal_length_deducted is None:
                    fail_manager.update_failed_id(image_id, "deduct_focal_length")
                    continue

                success = utie.update_table_images_estimated(image_id,
                                                             "focal_length", focal_length_deducted,
                                                             catch=catch, verbose=verbose, pbar=pbar)

                if success is False:
                    fail_manager.update_failed_id(image_id, "update_table_focal_length")
                    continue

    if debug_correct_focal_length:
        cfl.correct_focal_lengths("cam_id")

    # save the failed ids
    fail_manager.save_csv()

    p.print_v(f"Finished: estimate_for_all_images ({len(image_ids)} images)", verbose=verbose)


if __name__ == "__main__":

    img_ids = []

    if len(img_ids) == 0:
        _sql_string = "SELECT image_id FROM images"
        _data = ctd.get_data_from_db(_sql_string)
        img_ids = _data.values.tolist()
        img_ids = [item for sublist in img_ids for item in sublist]

    estimate_for_all_images(img_ids, overwrite=False, catch=False, verbose=True)
