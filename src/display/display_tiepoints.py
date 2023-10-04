import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.patches import ConnectionPatch
from tqdm import trange


def display_tiepoints(input_images, points,
                      confidences=None, filter_indices=None,
                      titles=None,
                      show_points=True, title=None,
                      save_path=None, verbose=False):

    # create a color between red and green based on a value between 0 and 1
    def pseudocolor(col_val, min_val, max_val, col_type):
        # Convert value in the range min_val...max_val to a color between red and green.
        fl = float(col_val - min_val) / (max_val - min_val)
        r, g, b = 1 - fl, fl, 0.
        if col_type == "rgb":
            r, g, b = round(r * 255.0), round(g * 255.0), round(b * 255.0)

        return r, g, b

    # plot the images
    f, axarr = plt.subplots(1, 2)
    if len(input_images[0].shape) == 2:
        axarr[0].imshow(input_images[0], cmap='gray')
    else:
        if input_images[0].shape[0] == 3:
            input_images[0] = np.moveaxis(input_images[0], 0, 2)
        axarr[0].imshow(input_images[0])
    if len(input_images[1].shape) == 2:
        axarr[1].imshow(input_images[1], cmap='gray')
    else:
        if input_images[1].shape[0] == 3:
            input_images[1] = np.moveaxis(input_images[1], 0, 2)
        axarr[1].imshow(input_images[1])

    if titles is not None:
        axarr[0].title.set_text(titles[0])
        axarr[1].title.set_text(titles[1])

    # if we don't want to filter we are creating a filter array in which we show everything
    if filter_indices is None:
        filter_indices = [1] * len(points)

    iterable = range(len(points))
    if verbose:
        iterable = trange(len(points))

    for i in iterable:

        # get the point row and confidence value
        point = points[i]
        if confidences is not None:
            conf = confidences[i]
        filtered = filter_indices[i]

        # get the left and right point
        p1 = (int(point[0]), int(point[1]))
        p2 = (int(point[2]), int(point[3]))

        if confidences is not None:
            col_rgba = pseudocolor(conf, 0, 1, "rgba")  # noqa
        else:
            col_rgba = "red"

        if show_points:

            if filtered == 1:
                axarr[0].scatter(p1[0], p1[1], color="b")
                axarr[1].scatter(p2[0], p2[1], color="b")
            else:
                axarr[0].scatter(p1[0], p1[1], color="r")
                axarr[1].scatter(p2[0], p2[1], color="r")

        if filtered == 1:
            con = ConnectionPatch(xyA=p1, coordsA=axarr[0].transData, xyB=p2,
                                  coordsB=axarr[1].transData, color=col_rgba,
                                  alpha=0.1)
            f.add_artist(con)

    if title is not None:
        f.suptitle(title)

    if save_path is None:
        plt.show()
    else:
        if save_path.endswith(".png") is False:
            save_path = save_path + ".png"
        plt.savefig(save_path, dpi=500)
        plt.clf()
        plt.close(f)  # close the current figure
