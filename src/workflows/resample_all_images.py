import tqdm as tqdm

import base.connect_to_db as ctd
import base.print_v as p

import image_resampling.calc_ppa as cp
import image_resampling.create_image_xml as cix
import image_resampling.resample_image as ri

def resample_all_images(image_ids, overwrite=False, catch=True, verbose=False):

    p.print_v(f"Start: resample_all_images({len(image_ids)} images)")

    for image_id in (pbar := tqdm(image_ids)):

        sql_string = f"SELECT * FROM images_fid_points WHERE image_id='{image_id}'"
        fid_data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        ppa = cp.calc_ppa(image_id, fid_data,
                          catch=catch, verbose=verbose, pbar=pbar)

