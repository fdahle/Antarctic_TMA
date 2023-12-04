import json
import os

import base.connect_to_db as ctd
import base.print_v as p


def find_overlapping_images(image_id, mode=None, jump=None,
                            catch=True, verbose=False, pbar=None):
    """
    find_overlapping_images(image_id, mode, jump, catch, verbose, pbar)
    Args:
        image_id (str): The image_id of the image for which we want to look for overlapping images
        mode (str): We can either look by 'image_id' or by 'approx_footprint'
        jump (integer): This is required for the mode 'image_id' and states how much the numbers of the image_id
            can be maximum apart for overlapping.
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar

    Returns:
        valid_ids (list): A list of ids that are overlapping with the image
    """

    p.print_v(f"Start: find_overlapping_images ({image_id})", verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if mode is None:
        mode = json_data["find_overlapping_images_mode"]

    if jump is None:
        jump = json_data["find_overlapping_images_jump"]

    try:
        if mode == "image_id":

            # split up the image_id
            flight = image_id[2:6]
            view_direction = image_id[8]
            frame = int(image_id[-3:])

            # with this sql-string we select all suitable image_ids
            sql_string = f"SELECT image_id, frame FROM images WHERE substring(image_id, 3, 4)='{flight}' " \
                         f"AND view_direction='{view_direction}'"
            data = ctd.get_data_from_db(sql_string, catch=catch, verbose=verbose, pbar=pbar)

            # get all rows with images close to the one we have
            valid_ids = data[(data['frame'] >= frame - jump) & (data['frame'] <= frame + jump)]
            valid_ids = valid_ids['image_id'].values.tolist()

        elif mode == "approx_footprint":
            raise NotImplementedError
        else:
            p.print_v("Please specify a valid mode", color="red")
            exit()

        # remove the image itself from the list
        valid_ids.remove(image_id)

    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: find_overlapping_images ({image_id})", verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v(f"Finished: find_overlapping_images ({image_id})", verbose, pbar=pbar)

    return valid_ids
