import copy

import cv2

import load_image_from_file as liff
import connect_to_db as ctd

"""remove_usgs_logo(img_path, logo_height, verbose, catch):
This function loads an image and removes the USGS logo from the bottom. Before removing it checks the database if
the logo is not already removed (to prevent data-loss)
INPUT:
    img_path (String): the path to the image from which the logo should be removed
    logo_height (Int, None): how much should be removed from the bottom part of the image (in pixels; normally 350)
    verbose (Boolean, False): If true, the status of the operations are printed
    catch (Boolean, True): If true and somethings is going wrong, the operation will continue and not crash.
OUTPUT:
    status: The status of the operation - True if a logo was removed (or if it doesn't need to be removed)

"""


def remove_usgs_logo(img_path, logo_height=None, verbose=False, catch=False):

    default_logo_height = 350

    # set default logo height
    if logo_height is None:
        logo_height = default_logo_height

    # get img_id from the path
    img_id = img_path.split("/")[-1][:-4]

    if verbose:
        print(f"Remove USGS logo from {img_id}")

    # load image
    img = liff.load_image_from_file(img_id)

    # save original for backup
    img_original = copy.deepcopy(img)

    # check if the usgs logo is still there
    sql_string = f"SELECT logo_removed FROM images_properties WHERE image_id ='{img_id}'"
    data = ctd.get_data_from_db(sql_string, verbose=verbose, catch=catch)
    status = data.iloc[0]['logo_removed']

    # the image still has its usgs logo
    if str(status) == "False":

        # remove the USGS logo
        img = img[0:img.shape[0] - logo_height, :]

        try:

            # save image
            cv2.imwrite(img_path, img)

            # save this information in the db
            sql_string = f"UPDATE images_properties SET logo_removed=TRUE WHERE image_id='{img_id}'"
            status = ctd.edit_data_in_db(sql_string, verbose=verbose, catch=False)

            # if this failed reverse the removing of USGS Logo
            if status is False:
                cv2.imwrite(img_path, img_original)
                if catch:
                    return False
                else:
                    raise Exception
            else:
                return True

        except (Exception,) as e:
            if catch:
                return False
            else:
                raise e
    else:
        return True
