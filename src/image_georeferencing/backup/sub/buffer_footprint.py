import json
import os

import base.print_v as p


def buffer_footprint(footprint, buffer_val=None,
                     catch=True, verbose=False, pbar=None):

    """
    buffer_footprint(approx_footprint, buffer_val, catch, verbose):
    This function buffers a approx_footprint on all side with a specified value. Note: This buffer function
    works with meters.
    Args:
        footprint (Shapely polygon): The polygon we want to buffer.
        buffer_val (Integer, None): The value we want to buffer the polygon on each side.
        catch (Boolean, True): If true and something is going wrong, the operation will continue and not crash.
            In this case None is returned
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        buffered_footprint (Shapely polygon): The buffered polygon
    """

    p.print_v("Start: buffer_footprint", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder[:-4] + "/params.json") as j_file:
        json_data = json.load(j_file)

    # set default buffer value if not specified
    if buffer_val is None:
        buffer_val = json_data["footprint_buffer"]

    try:
        buffered_footprint = footprint.buffer(buffer_val)
    except (Exception,) as e:
        p.print_v(f"Failed: buffer_footprint", verbose=verbose, color="red", pbar=pbar)
        if catch:
            return None
        else:
            raise e

    p.print_v("Finished: buffer_footprint", verbose=verbose, pbar=pbar)

    return buffered_footprint
