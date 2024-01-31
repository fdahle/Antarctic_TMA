import json
import os
import re

import base.print_v as p

debug_search_for_cam_id = False
debug_search_for_focal_length = False
debug_search_for_focal_length_regex = False
debug_search_for_lens_cone = True
debug_search_for_height = False


def extract_image_parameters(text_content, image_id,
                             min_height=None, max_height=None,
                             catch=True, verbose=False, pbar=None):
    """
    extract_image_parameters(text_content, image_id, search_for, catch, verbose, pbar):
    This function analyzes the extracted text from an image and looks for different image parameters
    based on certain rules.
    Args:
        text_content (List): The extracted text from an image (it's a list because text can be from
            different positions)
        image_id (String): The image-image_id of the for which we're extracting images
        min_height (int): When extracting height, the height must be at least this min value
        max_height (int): When extracting height, the height must be not bigger than this max value
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar

    Returns:

    """

    p.print_v(f"Start: extract_image_parameters ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if min_height is None:
        min_height = json_data["extract_image_parameters_min_height"]
    if max_height is None:
        max_height = json_data["extract_image_parameters_max_height"]

    # here we save the results
    result_dict = {
        "cam_id": None,
        "focal_length": None,
        "lens_cone": None,
        "height": None
    }

    text_per_box = text_content.split(";")

    try:
        # iterate all parts of the text
        for text_part in text_per_box:

            # search for cam image_id
            if debug_search_for_cam_id:

                p.print_v("Search for cam-image_id", verbose, pbar=pbar)

                pattern = r"5\d-[0-9]{3}"
                matches = re.findall(pattern, text_part)
                if len(matches) == 1:
                    result_dict["cam_id"] = matches[0]

            # search for focal length
            if debug_search_for_focal_length:

                p.print_v("Search for focal-length", verbose, pbar=pbar)

                # find all occurrences of 'mm' in text
                mm_positions = [m.start() for m in re.finditer("mm", text_part.lower())]

                for pos in mm_positions:

                    try:
                        left = max(0, pos - 8)
                        focal_part = text_part[left:pos]
                        focal_part_splits = focal_part.split(".")

                        first_part = focal_part_splits[-2][-3:]
                        second_part = focal_part_splits[-1][:3]

                        if len(first_part) == 3:
                            first_part = "1" + first_part[1:]
                            first_part = first_part[0] + "5" + first_part[2:]
                        elif len(first_part) == 2:
                            first_part = "15" + first_part[1:]
                        elif len(first_part) == 1:
                            first_part = "15" + first_part

                        focal_length = float(first_part + "." + second_part)
                        result_dict["focal_length"] = focal_length
                    except (Exception,):
                        pass

            if debug_search_for_focal_length_regex:

                try:
                    # no need to search if we already have a focal length
                    if result_dict["focal_length"] is None:

                        p.print_v("Search for focal-length regex", verbose, pbar=pbar)

                        pattern = r'5\d\.\d{2,3}\d?'
                        matches = re.findall(pattern, text_part)

                        if len(matches) == 1:
                            focal_length = float('1' + matches[0])
                            result_dict["focal_length"] = focal_length

                except (Exception,):
                    pass

            # search for lens cone
            if debug_search_for_lens_cone:

                p.print_v("Search for lens-cone", verbose, pbar=pbar)

                pattern = r"\b(?:DF|SF|KF|RF|DS)\d{3,4}\b"

                matches = re.findall(pattern, text_part)

                if len(matches) == 1:
                    result_dict["lens_cone"] = matches[0]

            # search for height
            if debug_search_for_height:

                p.print_v("Search for height", verbose, pbar=pbar)

                pattern = r"\b(" + "|".join([str(i) for i in range(min_height, max_height+1, 100)]) + r")\b"

                matches = re.findall(pattern, text_part)

                if len(matches) == 1:
                    result_dict["height"] = matches[0]

    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: extract_image_parameters ({image_id})", verbose=verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v(f"Finished: extract_image_parameters ({image_id})", verbose=verbose, pbar=pbar)

    return result_dict


if __name__ == "__main__":

    _image_id = "CA174131L0268"

    import base.connect_to_db as ctd
    sql_string = f"SELECT * FROM images_extracted WHERE image_id='{_image_id}'"
    data = ctd.get_data_from_db(sql_string)
    _text_content = data.iloc[0]['text_content']

    print("Text:")
    print(_text_content)

    results = extract_image_parameters(_text_content, _image_id)

    print("Results:")
    print(results)
