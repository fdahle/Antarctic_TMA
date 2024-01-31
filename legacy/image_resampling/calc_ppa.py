import base.print_v as p


def calc_ppa(image_id, data, catch=True, verbose=False, pbar=None):

    """
    calc_ppa(data, debug_show, catch):
    This function calculates the position of a ppa in a picture, based on the coordinates
    of four fiducial marks (one at every corner). The fiducial marks must be in a pandas
    dataframe with x and y as separate fields (fid_mark_1_x, fid_mark_1_y, ... , fid_mark_4_y)
    Args:
        image_id (str): The image_id from the image for which we are calculating the ppa
        data (pandas): Pandas frame with the coordinates of the fiducial marks
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        x (int): The x-coordinate of the calculated ppa
        y (int): The y coordinate of the calculated ppa
    """

    p.print_v(f"Start: calc_ppa ({image_id})", verbose=verbose, pbar=pbar)

    try:
        line1 = [[int(data["fid_mark_3_x"]), int(data["fid_mark_3_y"])],
                 [int(data["fid_mark_4_x"]), int(data["fid_mark_4_y"])]]

        line2 = [[int(data["fid_mark_2_x"]), int(data["fid_mark_2_y"])],
                 [int(data["fid_mark_1_x"]), int(data["fid_mark_1_y"])]]

    except (Exception,) as e:
        if catch:
            return None, None
        else:
            raise e

    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(x_diff, y_diff)
    if div == 0:
        if catch:
            return None, None
        else:
            raise Exception("lines do not intersect!")

    d = (det(*line1), det(*line2))
    x = int(det(d, x_diff) / div)
    y = int(det(d, y_diff) / div)

    p.print_v(f"Finished: calc_ppa ({image_id})", verbose=verbose, pbar=pbar)

    return x, y
