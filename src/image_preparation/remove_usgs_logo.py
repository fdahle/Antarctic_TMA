import copy
import cv2
import json
import os

import base.connect_to_db as ctd
import base.print_v as p


def remove_usgs_logo(image, image_id, logo_height=None, overwrite=True,
                     catch=True, verbose=False, pbar=None):

    """remove_usgs_logo(img_path, logo_height, verbose, catch):
    This function loads an image and removes the USGS logo from the bottom.
    Args:
        image (Np-array): The numpy array with the image
        image_id (String): The image_id of the image
        logo_height (Int, None): how much should be removed from the bottom part of the
            image (in pixels; normally 350)
        overwrite (Boolean): If true, we don't remove the logo if the logo is already removed
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar

    Returns:
        img_small: The image without a logo
    """

    p.print_v(f"Start: remove_usgs_logo ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    # we need the image (of course) but also the image_id
    assert image is not None, "No image could be found"
    assert image_id is not None, "Image-image_id is missing"

    # get the logo removed information from the db
    sql_string = f"SELECT path_file, logo_removed FROM images WHERE image_id='{image_id}'"
    data_image = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

    # check if we need to remove the usgs logo
    bool_logo = data_image["logo_removed"].iloc[0]
    if overwrite is False and str(bool_logo) == 'True':
        p.print_v(f"{image_id} already has logo removed", verbose, "green", pbar)
        return image

    # set default logo height
    if logo_height is None:
        logo_height = json_data["remove_usgs_logo_height_px"]

    p.print_v(f"Remove USGS logo from {image_id}", verbose, pbar=pbar)

    # copy the original image to not change it
    image = copy.deepcopy(image)

    try:
        # remove the USGS logo
        img_small = image[0:image.shape[0] - logo_height, :]

        # update the field in the database
        sql_string = f"UPDATE images SET logo_removed=True WHERE image_id='{image_id}'"
        success = ctd.edit_data_in_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

        # if something went wrong with updating we shouldn't save the new image
        if success is False:
            return None

        # save the new smaller image
        cv2.imwrite(data_image['path_file'].iloc[0], img_small)

        p.print_v(f"USGS logo removed successfully from {image_id}", verbose,
                  color="green", pbar=pbar)

    except (Exception,) as e:
        if catch:
            p.print_v(f"Something went wrong removing the logo from {image_id}", verbose,
                      color="red", pbar=pbar)
            return None
        else:
            raise e

    p.print_v(f"Finished: remove_usgs_logo ({image_id})", verbose, pbar=pbar)

    return img_small


if __name__ == "__main__":

    _image_id = "CA214732V0032"

    import base.load_image_from_file as liff
    img = liff.load_image_from_file(_image_id, catch=False)

    img = remove_usgs_logo(img, _image_id, overwrite=True, verbose=True)

    import display.display_images as di
    di.display_images(img)
