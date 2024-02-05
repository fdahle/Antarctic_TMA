import copy
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors

from matplotlib.colors import LinearSegmentedColormap


def display_images(images, types=None, titles=None,
                   title=None, x_titles=None, y_titles=None,
                   plot_shape=None,
                   points=None, lines=None, bboxes=None, polygons=None,
                   point_color=(255, 0, 0), point_size=7,
                   line_color=(255, 0, 0), line_width=1,
                   cmap_min=None, cmap_max=None,
                   bool_show_axis_marker=True,
                   save_path=None):
    """
    This function can be used to display images in one figure with some additional settings
    Args:
    - list_of_images: A list of images that should be displayed (usually a np-array)
    - list_of types: Optional a list of strings specifying the type of the images
         (changes how an image is displayed).
         Possible values are ["auto", "color", "gray", "segmented", "prob"]. If the value
         'auto' is selected, the program tries to determine the type itself
    - list_of_titles: Optional titles for the figures per image
    - title: A title for the complete plot
    - x_titles: titles for the columns
    - y_titles: titles for the rows
    - plot_shape: If none, the shape of all figures is determined automatically, otherwise
        it must be a tuple with (y, x)
    - points: A list of points that should be displayed in the images (x, y)
    - lines: A list of lines shat should be displayed between the images
        it must be in the format (x1, y1, x2, y2)
    - bboxes: A list of bboxes that should be displayed in the images (x, y, width, height)
    - cmap_min: the minimum value for a colormap
    - cmap_max: the maximum value for a colormap
    - bool_show_axis_marker: If false, the axis markers are removed
    - save_path: If a path is provided, the plot is saved there
    Returns:
        None
    """

    # check the images and make to list
    if type(images) is not list:
        images = [images]
    assert len(images) >= 1

    # check the types
    if types is not None:
        if type(types) is not list:
            types = [types]
    else:
        # if we don't have types we need to estimate the image type
        types = []
        for img in images:
            types.append(get_img_type(img))
    assert len(images) == len(types)

    # check the titles
    if titles is not None:
        if type(titles) is not list:
            titles = [titles]
        assert len(images) == len(titles)

    # check if we have points
    if points is not None:

        # usually if we only give one image, the points are also not a list
        if len(images) == 1:
            points = [points]

        assert len(points) == len(images)

    # check if we have lines
    if lines is not None:

        # usually if we only give one image, the lines are also not a list
        if len(images) == 1:
            lines = [lines]

        assert len(lines) == len(images)

    # check if we have bboxes
    if bboxes is not None:

        # usually if we only give one image, the bboxes are also not a list
        if len(bboxes) == 1:
            bboxes = [bboxes]

        assert len(bboxes) == len(images)

    # deepcopy the images so that we don't change them if we plot something
    for i, elem in enumerate(images):
        images[i] = copy.deepcopy(elem)

    # we need to figure out how to order the plots if it is not given
    if plot_shape is None:

        # we need the root for auto-calculation
        root = math.sqrt(len(images))

        # only one image -> that's easy
        if len(images) == 1:
            len_y, len_x = 1, 1

        # everything under 3 is just fine as a row
        elif len(images) <= 3:
            len_y, len_x = 1, len(images)

        # square numbers are easy
        elif int(root + 0.5) ** 2 == len(images):
            sq_num = int(math.sqrt(len(images)))
            len_y = sq_num
            len_x = sq_num

        # all the rest -> that's more difficult
        else:

            positions = get_good_squares(len(images))
            len_y = positions[0]
            len_x = positions[1]
    else:
        len_y = plot_shape[0]
        len_x = plot_shape[1]

    # assure that we can plot all images
    assert len_y * len_x == len(images)

    # create the plot
    f, axarr = plt.subplots(len_y, len_x, squeeze=False)

    # add title to the complete plot
    if title is not None:
        f.suptitle(title)

    # get colormap for probability
    c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
    v = [0, .15, .4, .5, 0.6, .9, 1.]
    cmap_prob = LinearSegmentedColormap.from_list('rg', list(zip(v, c)), N=256)

    # normalize colors
    point_color = normalize_color(point_color)
    line_color = normalize_color(line_color)

    # we need to keep track of the right image of the list
    img_counter = 0

    # iterate every position in axarr
    for y in range(len_y):
        for x in range(len_x):

            # check if points must be plotted
            if points is not None:

                # get the right points from the list
                points_img = points[img_counter]

                # iterate every point
                for pt in points_img:
                    axarr[y, x].scatter(pt[0], pt[1], color=point_color, s=point_size)

            # check if lines must be plotted
            if lines is not None:
                line_img = lines[img_counter]
                for ln in line_img:
                    x_values = [ln[0], ln[2]]
                    y_values = [ln[1], ln[3]]

                    axarr[y, x].plot(x_values, y_values, color=line_color, linewidth=line_width)

            # remove the axis markers if we don't want one
            if bool_show_axis_marker is False:
                axarr[y, x].set_yticklabels([])
                axarr[y, x].set_xticklabels([])
                axarr[y, x].tick_params(axis='both', which='both', length=0)

            # TODO FIND OUT WHAT THIS DOES
            # if img_counter >= len_lst:
            #    f.delaxes(axarr[y, x])
            #    continue

            # if we have titles for the columns and row, we need to display them
            if x == 0 and y_titles is not None:
                axarr[y, x].set_ylabel(y_titles[y])
            if y == 0 and x_titles is not None:
                axarr[y, x].set_xlabel(x_titles[x])
                axarr[y, x].xaxis.set_label_position('top')

            # get the actual image
            img = images[img_counter]
            img_type = types[img_counter]

            # fall back if we couldn't determine the image type -> display a black image
            if img_type is None:
                img = np.zeros((2, 2))

            # if we have a color image, we need the color axis at the end
            if img_type == "color" and img.shape[0] == 3:
                img = np.moveaxis(img, 0, 2)

            if img_type == "gray":
                axarr[y, x].imshow(img, cmap="gray", interpolation=None, vmin=0, vmax=255)
            elif img_type == "segmented":
                cmap, norm = create_cmap()
                axarr[y, x].imshow(img, cmap=cmap, norm=norm, interpolation=None)
            elif img_type == "color":
                axarr[y, x].imshow(img, interpolation=None)
            elif img_type == "binary":
                cmap_bin = matplotlib.colors.ListedColormap(['red', 'green'])
                axarr[y, x].imshow(img, cmap=cmap_bin)
            elif img_type == "prob":
                axarr[y, x].imshow(img, cmap=cmap_prob)
            elif img_type == "random":
                axarr[y, x].imshow(img, cmap=matplotlib.colors.ListedColormap(np.random.rand(256, 3)))
            elif img_type == "height":
                cmap = matplotlib.cm.get_cmap("RdYlGn_r")
                if cmap_min is not None and cmap_max is not None:
                    axarr[y, x].imshow(img, cmap=cmap, vmin=cmap_min, vmax=cmap_max)
                else:
                    axarr[y, x].imshow(img, cmap=cmap)

            # add titles to the single plots
            if titles is not None:
                axarr[y, x].title.set_text(titles[img_counter])

            # add bboxes to the plots
            if bboxes is not None:
                bbox = bboxes[img_counter]
                if bbox is not None and len(bbox) > 0:
                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                             linewidth=1, edgecolor='r', facecolor='none')
                    axarr[y, x].add_patch(rect)

            # add polygons to the plots
            if polygons is not None:
                poly = polygons[img_counter]
                if poly.geom_type == 'MultiPolygon':
                    for geom in poly.geoms:
                        xs, ys = geom.exterior.xy
                        axarr[y, x].fill(xs, ys, fc='none', ec="r")
                elif poly.geom_type == 'Polygon':
                    xs, ys = poly.exterior.xy
                    axarr[y, x].fill(xs, ys, fc='none', ec="r")

            img_counter += 1

    if save_path is None:
        plt.show()
    else:
        if save_path.endswith(".png") is False:
            save_path = save_path + ".png"
        plt.savefig(save_path, dpi=500)
        plt.clf()
        plt.close(f)  # close the current figure to prevent memory leak


# calculate how to plot a random number of data in the best format
def get_good_squares(num_images):

    root = int(math.sqrt(num_images))

    # if only one image -> easy
    if num_images == 1:
        return 1,

    # two images are easy
    elif num_images == 2:
        tpl = (1, 2)
        return tpl

    # three images can be put really easy together as well
    elif num_images == 3:
        tpl = (1, 3)
        return tpl

    # if it's a square number, also easy
    elif int(root + 0.5) ** 2 == num_images:

        tpl = (root, root)
        return tpl

    # not a square, there a solution must be found
    else:

        # function to get all divisors:
        def get_divisors(n):
            result = set()
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    result.add(i)
                    result.add(n // i)

            # divisors as list
            divs = list(result)

            # get the counterpart division
            both_divs = []
            for _elem in divs:
                both_divs.append((_elem, int(n / _elem)))

            return both_divs

        # get all divisors
        divisors = get_divisors(num_images)

        # if there's only 2 divisors often it's better to increase the number (11 sucks, 12 is good)
        if len(divisors) == 2:
            divisors = get_divisors(num_images + 1)

        # get difference between divisor
        diff_divisor = []
        for elem in divisors:
            diff_divisor.append(abs(elem[0] - elem[1]))

        # find the combination with the smallest difference between the divisors
        min_value = min(diff_divisor)
        min_idx = diff_divisor.index(min_value)
        min_pair = divisors[min_idx]

        return min_pair


# find out the type of image
def get_img_type(input_img):

    # check for 0 size and Athen stop checking -> otherwise error
    for elem in input_img.shape:
        if elem == 0:
            return "binary"

    # there's no image
    if input_img is None:
        _img_type = None
    elif len(input_img) == 0:
        _img_type = None

    # we have a gray scale image
    if len(input_img.shape) == 2:

        # image values are low, so special type of grayscale
        if np.nanmax(input_img) <= 1:
            # check if array is binary
            if np.array_equal(input_img, input_img.astype(bool)):
                _img_type = "binary"
            else:
                _img_type = "prob"

        # probably a grayscale
        elif np.nanmax(input_img) > 8:
            _img_type = "gray"

        # otherwise segmented
        else:
            _img_type = "segmented"

    elif len(input_img.shape) == 3:

        # check if the number of bands is at first position
        if input_img.shape[0] == 3:
            input_img = np.moveaxis(input_img, 0, 2)

        if input_img.shape[2] == 3:
            _img_type = "color"
        else:
            _img_type = "undefined"
    else:
        _img_type = "undefined"

    return _img_type


# create a colormap for segmentation
def create_cmap():
    color_dict = {
        "light_green": (102, 255, 0),  # for no data, 0
        "dark_gray": (150, 149, 158),  # ice, 1
        "light_gray": (230, 230, 235),  # snow, 2
        "black": (46, 45, 46),  # rocks, 3
        "dark_blue": (7, 29, 232),  # water, 4
        "light_blue": (25, 227, 224),  # clouds, 5
        "dark_red": (186, 39, 32),  # sky, 6
        "pink": (224, 7, 224)  # unknown, 7
    }

    # divide colors by 255 (important for matplotlib
    colors = []
    for elem in color_dict.values():
        col = tuple(ti / 255 for ti in elem)
        colors.append(col)

    custom_norm = matplotlib.colors.Normalize(vmin=0, vmax=len(color_dict.keys()))
    custom_cmap = matplotlib.colors.ListedColormap(colors)

    return custom_cmap, custom_norm


# normalize colors
def normalize_color(color):
    if all(isinstance(val, (int, float)) for val in color) and any(val > 1 for val in color):
        return tuple(val / 255. for val in color)
    return color