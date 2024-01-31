import os

import base.connect_to_db as ctd
import base.modify_shape as ms

def remove_image_by_id(img_id):

    user_input = input(f'Do you really want to delete {img_id}? (y/n): ')

    if user_input.lower() != 'y':
        return

    # reset in sql
    sql_string = "UPDATE images_georef SET calc_time=NULL, " \
                 "tps_sat=NULL, status_sat=NULL, tps_sat_est=NULL, status_sat_est=NULL, " \
                 "tps_img=NULL, status_img=NULL, tps_calc=NULL, status_calc=NULL, " \
                 "method=NULL, residuals=NULL, " \
                 "t0=NULL, t1=NULL, t2=NULL, t3=NULL, t4=NULL, t5=NULL, t6=NULL, t7=NULL, t8=NULL," \
                 "timestamp=NULL, pixel_size_x=NULL, pixel_size_y=NULL " \
                 f"WHERE image_id='{img_id}'"
    ctd.edit_data_in_db(sql_string, catch=False, add_timestamp=False)

    sql_string = "UPDATE images_extracted SET footprint_exact=NULL, position_exact=NULL, " \
                 "position_error_vector=NULL, footprint_type=NULL " \
                 f"WHERE image_id='{img_id}'"
    ctd.edit_data_in_db(sql_string, catch=False, add_timestamp=False)

    # remove from overview

    for type in ["sat", "img", "calc"]:

        # link to shp files
        footprints_shp = f"/data_1/ATM/data_1/playground/georef4/overview/{type}/{type}_footprints.shp"
        photocenters_shp = f"/data_1/ATM/data_1/playground/georef4/overview/{type}/{type}_photocenters.shp"
        error_vectors_shp = f"/data_1/ATM/data_1/playground/georef4/overview/{type}/{type}_error_vectors.shp"

        footprints_invalid_shp = f"/data_1/ATM/data_1/playground/georef4/overview/{type}_invalid/{type}_footprints_invalid.shp"
        photocenters_invalid_shp = f"/data_1/ATM/data_1/playground/georef4/overview/{type}_invalid/{type}_photocenters_invalid.shp"
        error_vectors_invalid_shp = f"/data_1/ATM/data_1/playground/georef4/overview/{type}_invalid/{type}_error_vectors_invalid.shp"

        if os.path.isfile(footprints_shp):
            ms.modify_shape(footprints_shp, img_id, "delete")
        if os.path.isfile(photocenters_shp):
            ms.modify_shape(photocenters_shp, img_id, "delete")
        if os.path.isfile(error_vectors_shp):
            ms.modify_shape(error_vectors_shp, img_id, "delete")

        if os.path.isfile(footprints_invalid_shp):
            ms.modify_shape(footprints_invalid_shp, img_id, "delete")
        if os.path.isfile(photocenters_invalid_shp):
            ms.modify_shape(photocenters_invalid_shp, img_id, "delete")
        if os.path.isfile(error_vectors_invalid_shp):
            ms.modify_shape(error_vectors_invalid_shp, img_id, "delete")


    # remove from tiffs
    for type in ["sat", "sat_invalid", "img", "img_invalid", "calc", "calc_invalid"]:
        tiff_path = f"/data_1/ATM/data_1/playground/georef4/tiffs/{type}/{img_id}.tif"
        if os.path.isfile(tiff_path):
            os.remove(tiff_path)

    #remove from tie-points images
    for type in ["sat_too_few_tps", "sat_invalid", "sat"]:
        png_path = f"/data_1/ATM/data_1/playground/georef4/tie_points/images/{type}/{img_id}_points.png"
        if os.path.isfile(png_path):
            os.remove(png_path)

    #remove from tie-points shapes
    for type in ["sat_too_few_tps", "sat_invalid", "sat", "img_too_few_tps", "img_invalid", "img"]:
        shp_path = f"/data_1/ATM/data_1/playground/georef4/tie_points/shapes/{type}/{img_id}_points.shp"
        if os.path.isfile(shp_path):
            ms.modify_shape(shp_path, img_id, "hard_delete")

    # remove low res
    low_res_path = f"/data_1/ATM/data_1/playground/georef4/low_res/images/{img_id}.tif"
    if os.path.isfile(low_res_path):
        os.remove(low_res_path)

if __name__ == "__main__":
    remove_image_by_id("CA183132V0002")