import os

from tqdm import tqdm

import sub.check_georef_image as cgi
import sub.get_invalid_position as gip

import base.connect_to_db as ctd
import base.load_shape_data as lsd
import base.modify_shape as ms
import base.print_v as p

# here we want to store the created data
base_path = "/data_1/ATM/data_1/playground/georef3"

debug_print_all = True
debug_relocate = True
debug_check_shapes = True


def clean_georef_folder(image_id=None, test_valid=True, test_invalid=True, mode="sat",
                        verbose=False):

    # overview paths
    path_exact_footprints_shape_valid = base_path + f"/overview/{mode}/{mode}_footprints.shp"
    path_exact_footprints_shape_invalid = base_path + f"/overview/{mode}_invalid/{mode}_footprints_invalid.shp"
    path_exact_photocenters_shape_valid = base_path + f"/overview/{mode}/{mode}_photocenters.shp"
    path_exact_photocenters_shape_invalid = base_path + f"/overview/{mode}_invalid/{mode}_photocenters_invalid.shp"
    path_error_vector_shape_valid = base_path + f"/overview/{mode}/{mode}_error_vectors.shp"
    path_error_vector_shape_invalid = base_path + f"/overview/{mode}_invalid/{mode}_error_vectors_invalid.shp"

    # load the data
    footprints_data_valid = lsd.load_shape_data(path_exact_footprints_shape_valid)
    footprints_data_invalid = lsd.load_shape_data(path_exact_footprints_shape_invalid)
    photocenters_data_valid = lsd.load_shape_data(path_exact_photocenters_shape_valid)
    photocenters_data_invalid = lsd.load_shape_data(path_exact_photocenters_shape_invalid)
    error_vectors_data_valid = lsd.load_shape_data(path_error_vector_shape_valid)
    error_vectors_data_invalid = lsd.load_shape_data(path_error_vector_shape_invalid)

    if mode == "sat":
        sql_mode = "satellite"

    # we want to check all images
    if image_id is None:
        if test_valid:
            image_ids_valid = os.listdir(base_path + f"/tiffs/{mode}/")
            image_ids_valid = [x.split('.')[0] for x in image_ids_valid]
        else:
            image_ids_valid = []

        if test_invalid:
            image_ids_invalid = os.listdir(base_path + f"/tiffs/{mode}_invalid/")
            image_ids_invalid = [x.split('.')[0] for x in image_ids_invalid]
        else:
            image_ids_invalid = []

    # we want to check only one image
    else:
        image_ids_valid = [image_id]
        image_ids_invalid = [image_id]

    # to store the changed ids
    new_valid_ids = []
    new_invalid_ids = []

    # iterate all valid images
    for img_id in (pbar := tqdm(image_ids_valid)):

        if debug_print_all:
            pbar = None

        p.print_v(f"Check valid image '{img_id}'", verbose=verbose, pbar=pbar)

        # prepare data for overview shapes
        footprint_data = footprints_data_valid[footprints_data_valid['image_id'] == img_id]
        if len(footprint_data) == 1:
            footprint_wkt = footprint_data['geometry'].iloc[0].wkt
            footprint_data = footprint_data.drop(['image_id', 'geometry'], axis=1)
            footprint_data = footprint_data.to_dict(orient='records')[0]

        photocenter_data = photocenters_data_valid[photocenters_data_valid['image_id'] == img_id]
        if len(photocenter_data) == 1:
            photocenter_wkt = photocenter_data['geometry'].iloc[0].wkt

        error_vector_data = error_vectors_data_valid[error_vectors_data_valid['image_id'] == img_id]
        if len(error_vector_data) == 1:
            error_vector_wkt = error_vector_data['geometry'].iloc[0].wkt
            error_vector_data = error_vector_data.drop(['image_id', 'geometry'], axis=1)
            error_vector_data = error_vector_data.to_dict(orient='records')[0]

        # tiff paths
        path_tiff_valid = base_path + f"/tiffs/{mode}/{img_id}.tif"
        path_tiff_invalid = base_path + f"/tiffs/{mode}_invalid/{img_id}.tif"

        # tie point paths
        path_tie_points_img_valid = base_path + f"/tie_points/images/{mode}/{img_id}.png"
        path_tie_points_img_invalid = base_path + f"/tie_points/images/{mode}_invalid/{img_id}.png"
        path_tie_points_shp_valid = base_path + f"/tie_points/shapes/{mode}/{img_id}_points.shp"
        path_tie_points_shp_invalid = base_path + f"/tie_points/shapes/{mode}_invalid/{img_id}_points.shp"

        # first just check if the image itself is valid
        try:
            validity_image = cgi.check_georef_image(path_tiff_valid, verbose=verbose)
        except (Exception,):
            validity_image = False

        # now check if the position of the  image is not an outlier
        validity_position = gip.get_invalid_position(img_id)

        if validity_image is False or validity_position is False:

            p.print_v(f"Image is invalid (Position: {validity_position}, Image: {validity_image})",
                      color="red", verbose=verbose, pbar=pbar)

            new_invalid_ids.append(img_id)

            if debug_relocate:

                # relocate images to the invalid folder
                os.rename(path_tiff_valid, path_tiff_invalid) if os.path.exists(path_tiff_valid) else None

                # relocate tie point images to the invalid folder
                os.rename(path_tie_points_img_valid, path_tie_points_img_invalid) if os.path.exists(
                    path_tie_points_img_valid) \
                    else None

                # we need to make the path shorter, as we cannot move only the shp file but others as well
                short_path = path_tie_points_shp_valid[:-3]
                short_path_invalid = path_tie_points_shp_invalid[:-3]

                # relocate all parts of the tie point shape to the invalid folder
                os.rename(short_path + "cpg", short_path_invalid + "cpg") if os.path.exists(
                    short_path + "cpg") else None
                os.rename(short_path + "dbf", short_path_invalid + "dbf") if os.path.exists(
                    short_path + "dbf") else None
                os.rename(short_path + "prj", short_path_invalid + "prj") if os.path.exists(
                    short_path + "prj") else None
                os.rename(short_path + "shp", short_path_invalid + "shp") if os.path.exists(
                    short_path + "shp") else None
                os.rename(short_path + "shx", short_path_invalid + "shx") if os.path.exists(
                    short_path + "shx") else None

                # relocate overview
                if len(footprint_data) == 1:
                    ms.modify_shape(path_exact_footprints_shape_invalid, img_id, "add",
                                    geometry_data=footprint_wkt, optional_data=footprint_data)
                    ms.modify_shape(path_exact_footprints_shape_valid, img_id, "delete")
                if len(photocenter_data) == 1:
                    ms.modify_shape(path_exact_photocenters_shape_invalid, img_id, "add",
                                    geometry_data=photocenter_wkt)
                    ms.modify_shape(path_exact_photocenters_shape_valid, img_id, "delete")
                if len(error_vector_data) == 1:
                    ms.modify_shape(path_error_vector_shape_invalid, img_id, "add",
                                    geometry_data=error_vector_wkt, optional_data=error_vector_data)
                    ms.modify_shape(path_error_vector_shape_valid, img_id, "delete")

                # update also sql
                sql_string = "UPDATE image_extracted SET footprint = NULL, position_exact = NULL, " \
                             "position_error_vector = NULL, footprint_type = NULL " \
                             f"WHERE image_id='{img_id}' AND footprint_type='{sql_mode}'"
                ctd.edit_data_in_db(sql_string)
        else:
            p.print_v(f"Image is valid", color="green", verbose=verbose, pbar=pbar)

    print(f"{len(new_invalid_ids)} images of {len(image_ids_valid)} valid images are moved to invalid")

    # iterate all invalid images
    for img_id in (pbar := tqdm(image_ids_invalid)):

        if debug_print_all:
            pbar = None

        p.print_v(f"Check invalid image '{img_id}'", verbose=verbose, pbar=pbar)

        # prepare data for overview shapes
        footprint_data = footprints_data_invalid[footprints_data_invalid['image_id'] == img_id]
        if len(footprint_data) == 1:
            footprint_wkt = footprint_data['geometry'].iloc[0].wkt
            footprint_data = footprint_data.drop(['image_id', 'geometry'], axis=1)
            footprint_data = footprint_data.to_dict(orient='records')[0]

        photocenter_data = photocenters_data_invalid[photocenters_data_invalid['image_id'] == img_id]
        if len(photocenter_data) == 1:
            photocenter_wkt = photocenter_data['geometry'].iloc[0].wkt

        error_vector_data = error_vectors_data_invalid[error_vectors_data_invalid['image_id'] == img_id]
        if len(error_vector_data) == 1:
            error_vector_wkt = error_vector_data['geometry'].iloc[0].wkt
            error_vector_data = error_vector_data.drop(['image_id', 'geometry'], axis=1)
            error_vector_data = error_vector_data.to_dict(orient='records')[0]

        # tiff paths
        path_tiff_valid = base_path + f"/tiffs/{mode}/{img_id}.tif"
        path_tiff_invalid = base_path + f"/tiffs/{mode}_invalid/{img_id}.tif"

        # tie point paths
        path_tie_points_img_valid = base_path + f"/tie_points/images/{mode}/{img_id}.png"
        path_tie_points_img_invalid = base_path + f"/tie_points/images/{mode}_invalid/{img_id}.png"
        path_tie_points_shp_valid = base_path + f"/tie_points/shapes/{mode}/{img_id}_points.shp"
        path_tie_points_shp_invalid = base_path + f"/tie_points/shapes/{mode}_invalid/{img_id}_points.shp"

        # first just check if the image itself is valid
        try:
            validity_image = cgi.check_georef_image(path_tiff_invalid, verbose=verbose)
        except (Exception,):
            validity_image = False

        # now check if the position of the  image is not an outlier
        validity_position = gip.get_invalid_position(img_id)

        if validity_image is True and validity_position is True:

            p.print_v("Image is valid", color="green", verbose=verbose, pbar=pbar)

            new_valid_ids.append(img_id)

            if debug_relocate:

                # relocate images to the invalid folder
                os.rename(path_tiff_invalid, path_tiff_valid) if os.path.exists(path_tiff_invalid) else None

                # relocate tie point images to the invalid folder
                os.rename(path_tie_points_img_invalid, path_tie_points_img_valid) if os.path.exists(
                    path_tie_points_img_invalid) \
                    else None

                # we need to make the path shorter, as we cannot move only the shp file but others as well
                short_path_valid = path_tie_points_shp_valid[:-3]
                short_path_invalid = path_tie_points_shp_invalid[:-3]

                # relocate all parts of the tie point shape to the invalid folder
                os.rename(short_path_invalid + "cpg", short_path_valid + "cpg") if os.path.exists(
                    short_path_invalid + "cpg") else None
                os.rename(short_path_invalid + "dbf", short_path_valid + "dbf") if os.path.exists(
                    short_path_invalid + "dbf") else None
                os.rename(short_path_invalid + "prj", short_path_valid + "prj") if os.path.exists(
                    short_path_invalid + "prj") else None
                os.rename(short_path_invalid + "shp", short_path_valid + "shp") if os.path.exists(
                    short_path_invalid + "shp") else None
                os.rename(short_path_invalid + "shx", short_path_valid + "shx") if os.path.exists(
                    short_path_invalid + "shx") else None

                # relocate overview
                if len(footprint_data) == 1:
                    ms.modify_shape(path_exact_footprints_shape_valid, img_id, "add",
                                    geometry_data=footprint_wkt, optional_data=footprint_data)
                    ms.modify_shape(path_exact_footprints_shape_invalid, img_id, "delete")
                if len(photocenter_data) == 1:
                    ms.modify_shape(path_exact_photocenters_shape_valid, img_id, "add",
                                    geometry_data=photocenter_wkt)
                    ms.modify_shape(path_exact_photocenters_shape_invalid, img_id, "delete")
                if len(error_vector_data) == 1:
                    ms.modify_shape(path_error_vector_shape_valid, img_id, "add",
                                    geometry_data=error_vector_wkt, optional_data=error_vector_data)
                    ms.modify_shape(path_error_vector_shape_invalid, img_id, "delete")

                # update also sql
                sql_string = "UPDATE image_extracted SET footprint = NULL, position_exact = NULL, " \
                             "position_error_vector = NULL, footprint_type = NULL " \
                             f"WHERE image_id='{img_id}' AND footprint_type='{sql_mode}'"
                ctd.edit_data_in_db(sql_string)
        else:
            p.print_v(f"Image is invalid (Position: {validity_position}, Image: {validity_image})",
                      color="red", verbose=verbose, pbar=pbar)

    print(f"{len(new_valid_ids)} images of {len(image_ids_invalid)} invalid images are moved to valid")

    # check if we have images in the shapefiles that are not any longer in the fld
    if debug_check_shapes:

        if test_valid:

            # load data again (because we deleted stuff)
            footprints_data = lsd.load_shape_data(path_exact_footprints_shape_valid)
            photocenters_data = lsd.load_shape_data(path_exact_photocenters_shape_valid)
            error_vectors_data = lsd.load_shape_data(path_error_vector_shape_valid)

            valid_ids_to_delete = []

            # iterate all image ids
            for _, row in footprints_data.iterrows():

                img_id = row['image_id']
                path_tiff_valid = base_path + f"/tiffs/{mode}/{img_id}.tif"

                # check if id still exists as an image
                if os.path.exists(path_tiff_valid) is False:
                    valid_ids_to_delete.append(img_id)

            for _, row in photocenters_data.iterrows():

                img_id = row['image_id']
                path_tiff_valid = base_path + f"/tiffs/{mode}/{img_id}.tif"

                # check if id still exists as an image
                if os.path.exists(path_tiff_valid) is False:
                    valid_ids_to_delete.append(img_id)

            for _, row in error_vectors_data.iterrows():

                img_id = row['image_id']
                path_tiff_valid = base_path + f"/tiffs/{mode}/{img_id}.tif"

                # check if id still exists as an image
                if os.path.exists(path_tiff_valid) is False:
                    valid_ids_to_delete.append(img_id)

            valid_ids_to_delete = set(valid_ids_to_delete)

            # delete from shape
            for img_id in valid_ids_to_delete:
                ms.modify_shape(path_exact_footprints_shape_valid, img_id, "delete")
                ms.modify_shape(path_exact_photocenters_shape_valid, img_id, "delete")
                ms.modify_shape(path_error_vector_shape_valid, img_id, "delete")
                sql_string = "UPDATE image_extracted SET footprint = NULL, position_exact = NULL, " \
                             "position_error_vector = NULL, footprint_type = NULL " \
                             f"WHERE image_id='{img_id}' AND footprint_type='{sql_mode}'"
                ctd.edit_data_in_db(sql_string)

            print(f"{len(valid_ids_to_delete)} ids are deleted from the valid shapes as the picture are missing")

        if test_invalid:

            # load data again (because we deleted stuff)
            footprints_data = lsd.load_shape_data(path_exact_footprints_shape_invalid)
            photocenters_data = lsd.load_shape_data(path_exact_photocenters_shape_invalid)
            error_vectors_data = lsd.load_shape_data(path_error_vector_shape_invalid)

            invalid_ids_to_delete = []

            # iterate all image ids
            for _, row in footprints_data.iterrows():

                img_id = row['image_id']
                path_tiff_invalid = base_path + f"/tiffs/{mode}_invalid/{img_id}.tif"

                # check if id still exists as an image
                if os.path.exists(path_tiff_invalid) is False:
                    invalid_ids_to_delete.append(img_id)

            for _, row in photocenters_data.iterrows():

                img_id = row['image_id']
                path_tiff_invalid = base_path + f"/tiffs/{mode}_invalid/{img_id}.tif"

                # check if id still exists as an image
                if os.path.exists(path_tiff_invalid) is False:
                    invalid_ids_to_delete.append(img_id)

            for _, row in error_vectors_data.iterrows():

                img_id = row['image_id']
                path_tiff_invalid = base_path + f"/tiffs/{mode}_invalid/{img_id}.tif"

                # check if id still exists as an image
                if os.path.exists(path_tiff_invalid) is False:
                    invalid_ids_to_delete.append(img_id)

            invalid_ids_to_delete = set(invalid_ids_to_delete)

            # delete from shape and database
            for img_id in invalid_ids_to_delete:
                ms.modify_shape(path_exact_footprints_shape_invalid, img_id, "delete")
                ms.modify_shape(path_exact_photocenters_shape_invalid, img_id, "delete")
                ms.modify_shape(path_error_vector_shape_invalid, img_id, "delete")

            print(f"{len(invalid_ids_to_delete)} ids are deleted from the invalid shapes as the picture are missing")


if __name__ == "__main__":

    img_id = None

    clean_georef_folder(image_id=img_id,verbose=True)
