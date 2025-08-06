"""show images with matplotlib"""

import copy
import math
import matplotlib

matplotlib.use('TkAgg')  # noqa: E402
import matplotlib.colors as mcolors  # noqa: SpellCheckingInspection
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Polygon
from shapely.geometry.polygon import Polygon as ShapelyPolygon
from typing import List, Tuple, Optional, Dict, Any, Union

base_style_config = {
    "axis_marker": True,
    "overlay_alpha": 0.5,
    "point_color": (255, 0, 0),
    "point_size": 7,
    "polygon_color": (255, 0, 0),
    "polygon_line_width": 1,
    "plot_shape": "auto",
    "line_color": (255, 0, 0),
    "line_width": 1,
    "title": None,
    "titles_sup": None,
    "titles_x": None,
    "titles_y": None
}


def display_images(images: np.ndarray | list[np.ndarray],
                   image_types: str | list[str] | None = None,
                   overlays: np.ndarray | List[np.ndarray] | None = None,
                   points: Optional[List[List[Tuple[int, int]]]] = None,
                   lines: Optional[List[List[Tuple[int, int, int, int]]]] = None,
                   bounding_boxes: Optional[List[List[Union[Tuple[int, int, int, int], List[int]]]]] = None,
                   polygons: Optional[List[List[ShapelyPolygon]]] = None,
                   polygons_text: Optional[List[List[str]]] = None,
                   polygons_color=None,
                   tie_points: Optional[np.ndarray] = None,
                   tie_points_conf: List[float] | np.ndarray | None = None,
                   reduce_tie_points: bool = False,
                   num_reduced_tie_points: int = 100,
                   separate_min_max: bool = False,
                   style_config: Dict[str, Any] | None = None,
                   save_path: str | None = None,
                   save_type: str = "png") -> None:
    """
    Displays or saves a series of images with optional annotations including points, lines, and tie-points.

    Args:
        images (List[np.ndarray]): A list of images to display.
        image_types (Optional[List[str]], optional): A list of strings indicating the type of each image.
        overlays (Optional[List[np.ndarray]], optional): A list of overlay images to display on top of the main images.
        points (Optional[List[List[Tuple[int, int]]]], optional): Points to mark on the images. Defaults to None.
            Expected in (x,y)
        lines (Optional[List[List[Tuple[int, int, int, int]]]], optional): Lines to draw on the images.
            Defaults to None.
        bounding_boxes (Optional[List[List[Tuple[int, int, int, int]]]], optional): Bounding boxes to
            draw on the images. Each bounding box is represented as a tuple (x_min, y_min, x_max, y_max).
            Defaults to None.
        polygons (Optional[List[List[ShapelyPolygon]]], optional): Polygons to draw on the images. Defaults to None.
        polygons_text (Optional[List[List[str]]], optional): Text to display inside the polygons. Defaults to None.
        polygons_color (Optional[List[Tuple[int, int, int]]], optional): Colors for the polygons. Defaults to None.
        tie_points (Optional[np.ndarray], optional): Array of tie-points connecting two images. Defaults to None.
        tie_points_conf (Optional[List[float]], optional): Confidence values for tie-points, affecting their appearance.
            Defaults to None.
        reduce_tie_points (bool, optional): Whether to reduce the number of tie-points displayed. Defaults to False.
        num_reduced_tie_points (int, optional): Number of tie-points to display if `reduce_tie_points` is True.
        style_config (Optional[Dict[str, Any]], optional): Configuration for styling the annotations. Defaults to None.
        save_path (Optional[str], optional): Path to save the figure instead of displaying. Defaults to None.
        save_type (str, optional): The file format for saving the figure ('png' or 'svg'). Defaults to "png".

    Raises:
        ValueError: If `save_type` is not supported or if `tie_points` format is incorrect.
    """

    # images must always be in a list
    if not isinstance(images, list):
        images = [images]

    # we need at least one image
    assert len(images) > 0, "No images to display."

    # validate the images
    for img in images:
        if not isinstance(img, np.ndarray):
            raise ValueError("All input images must be numpy arrays.")

    # Validate tie_points and adjust for tie-points scenario
    if tie_points is not None:
        assert tie_points.ndim == 2 and tie_points.shape[1] == 4, \
            "tie_points must be a 2D numpy array with shape (x, 4)"
        assert len(images) == 2, \
            "Exactly two images must be provided for tie-points."

    # If style_config is None, make it an empty dict to avoid TypeError when unpacking
    if style_config is None:
        style_config = {}

    # Merge the user's style_config with the base_style_config
    # The user's style_config will override any default settings if specified
    style_config = {**base_style_config, **style_config}

    # determine plot shape
    if tie_points is not None:
        plot_shape = (1, 2)
    elif style_config['plot_shape'] == "auto":
        plot_shape = _determine_plot_shape(len(images))
    else:
        plot_shape = style_config['plot_shape']

    # reduce tie-points if specified
    if reduce_tie_points and tie_points is not None:
        # random sample rows from tie_points and tie_points_conf
        idx = np.random.choice(tie_points.shape[0], num_reduced_tie_points, replace=False)
        new_tps = tie_points[idx]
        if tie_points_conf is not None:
            new_conf = [tie_points_conf[i] for i in idx]
    else:
        new_tps = tie_points
        new_conf = tie_points_conf

    # Calculate global cmin and cmax for all DEM images
    if separate_min_max is False:
        dem_min = np.inf
        dem_max = -np.inf
        if image_types is not None:
            for img, img_type in zip(images, image_types):
                if img_type == "dem":
                    dem_min = min(dem_min, np.nanmin(img))
                    dem_max = max(dem_max, np.nanmax(img))
        img_min_vals = [dem_min] * len(images)
        img_max_vals = [dem_max] * len(images)
    if separate_min_max:
        img_min_vals = []
        img_max_vals = []
        for img, img_type in zip(images, image_types):
            if img_type == "dem":
                img_min_vals.append(np.nanmin(img))
                img_max_vals.append(np.nanmax(img))
            else:
                img_min_vals.append(None)
                img_max_vals.append(None)

    # create the plot
    fig, axarr = plt.subplots(plot_shape[0], plot_shape[1], squeeze=False)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust space between plots

    # Set the figure's main title if specified
    if style_config['title']:
        fig.suptitle(style_config['title'])

    # Flatten axarr to simplify iterating over it if it's multidimensional
    axes = axarr.flatten()

    # define colormaps
    c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
    v = [0, .15, .4, .5, 0.6, .9, 1.]
    ls = list(zip(v, c))
    cmap_red_green = LinearSegmentedColormap.from_list('rg', ls, N=256)  # noqa

    c = ["darkgreen", "green", "palegreen", "white", "lightcoral", "red", "darkred"]
    v = [0, .15, .4, .5, 0.6, .9, 1.]
    ls = list(zip(v, c))
    cmap_green_red = LinearSegmentedColormap.from_list('rg', ls, N=256)  # noqa

    # Iterate over images and their corresponding axes
    for idx, enum_img in enumerate(images):

        # deep copy image to not change it
        img = copy.deepcopy(enum_img)
        img = np.asarray(img)

        # get ax for that image
        ax = axes[idx]

        # determine the type of the image
        if image_types is None:
            img_type = _determine_image_type(img)  # noqa
        else:
            img_type = image_types[idx]

        # we need to assure color images have the right format
        if img_type == "color" and img.shape[0] == 3:
            img = np.moveaxis(img, 0, 2)

        # show image differently based on the image type
        if img_type == "gray":
            ax.imshow(img, cmap="gray")
        elif img_type == "dem":
            dem_min = img_min_vals[idx]  # noqa
            dem_max = img_max_vals[idx]  # noqa
            ax.imshow(img, cmap="terrain", vmin=dem_min, vmax=dem_max)
        elif img_type == "color":
            ax.imshow(img, interpolation=None)
        elif img_type == "binary":
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        elif img_type == "rtg":  # red to green
            ax.imshow(img, cmap=cmap_red_green, vmin=np.nanmin(img), vmax=np.nanmax(img))
        elif img_type == "gtr":  # green to red
            ax.imshow(img, cmap=cmap_green_red)
        elif img_type == "difference":
            ax.imshow(img, cmap="seismic", vmin=np.nanmin(img), vmax=np.nanmax(img))
        elif img_type == "segmented":
            cmap, norm = _create_segmented_cmap()
            ax.imshow(img, cmap=cmap, norm=norm, interpolation=None)
        else:  # Default or undefined types
            ax.imshow(img)

        # Optionally draw overlay image
        if overlays is not None and idx < len(overlays):
            if overlays[idx] is not None:
                overlay_img = overlays[idx]

                if np.amax(overlay_img) == 1:
                    o_cmap = cmap_red_green
                else:
                    o_cmap = "gray"

                ax.imshow(overlay_img, cmap=o_cmap,
                          alpha=style_config['overlay_alpha'])

        # Optionally draw tie-points
        if tie_points is not None and idx == 0:  # Draw tie-points only from the context of the first image
            for tp_idx, point in enumerate(new_tps):
                p1 = (point[0], point[1])  # Coordinates in the first image
                p2 = (point[2], point[3])  # Corresponding coordinates in the second image

                # Determine color for tie points
                if tie_points_conf is not None:
                    conf = new_conf[tp_idx]  # noqa
                    # Interpolate color from red (low confidence) to green (high confidence)
                    color = (1 - conf, conf, 0)  # Red to green, based on conf
                else:
                    color = _normalize_color(style_config['line_color'])

                # Create and add ConnectionPatch with RGBA color
                con = ConnectionPatch(xyA=p1, coordsA=axes[0].transData, xyB=p2,
                                      coordsB=axes[1].transData,
                                      color=color,  # Append alpha to normalized color
                                      linewidth=style_config['line_width'],
                                      zorder=1)
                fig.add_artist(con)

        # Optionally draw points on the image
        if points is not None and idx < len(points):
            for point in points[idx]:
                ax.plot(point[0], point[1], 'o',
                        color=_normalize_color(style_config['point_color']),
                        markersize=style_config['point_size'])

        # Optionally draw lines on the image
        if lines and idx < len(lines):

            # get the lines for the current image
            image_lines = lines[idx]

            if len(image_lines) >= 1:

                # get the line colors
                line_colors = style_config['line_color']

                # Replicate single color if only one provided
                if not isinstance(line_colors, list):
                    line_colors = [line_colors] * len(image_lines)
                else:
                    line_colors = [line_colors[idx]] * len(image_lines)

                # plot each line with its corresponding color
                for line, color in zip(image_lines, line_colors):
                    if len(line) == 1:
                        line = line[0]  # Unpack single line tuple if needed
                    try:
                        ax.plot([line[0], line[2]], [line[1], line[3]],
                            color=_normalize_color(color),
                            linewidth=style_config['line_width'],
                            zorder=5)
                    except Exception as e:
                        raise e

        # Optionally draw bounding boxes on the image
        if bounding_boxes and idx < len(bounding_boxes):
            for bbox in bounding_boxes[idx]:
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=style_config['polygon_line_width'],
                                     edgecolor=_normalize_color(style_config['polygon_color']),
                                     facecolor='none')
                ax.add_patch(rect)

        # Optionally draw polygons on the image
        if polygons and idx < len(polygons):
            for poly_idx, poly in enumerate(polygons[idx]):
                if not isinstance(poly, ShapelyPolygon):
                    print("Warning: Skipping invalid polygon.")
                    continue  # Skip if not a ShapelyPolygon

                # Get the exterior coordinates of the polygon
                exterior_coords = np.array(poly.exterior.coords)

                # determine polygon color
                if polygons_color is None:
                    # get default color
                    poly_color = _normalize_color(style_config['polygon_color'])
                else:
                    poly_color = _normalize_color(polygons_color[poly_idx])

                # ignore empty polygons (0 width or height)
                if (np.amin(exterior_coords[:, 0]) == np.amax(exterior_coords[:, 0]) or
                        np.amin(exterior_coords[:, 1]) == np.amax(exterior_coords[:, 1])):
                    continue

                polygon_patch = Polygon(exterior_coords,
                                        linewidth=style_config['polygon_line_width'],
                                        edgecolor=poly_color,
                                        facecolor='none',
                                        zorder=50)
                ax.add_patch(polygon_patch)

                # Optionally add text to the polygon
                if polygons_text and idx < len(polygons_text) and poly_idx < len(polygons_text[idx]):
                    # Calculate the centroid of the polygon
                    centroid = poly.centroid

                    text = polygons_text[idx][poly_idx]
                    ax.text(centroid.x, centroid.y, text,
                            ha='center', va='center',
                            fontsize=14, color='black',
                            zorder=100)

        # Set subtitles for each subplot if specified
        if style_config.get('titles_sup') and idx < len(style_config['titles_sup']):
            ax.set_title(style_config['titles_sup'][idx], fontsize=10)

        # Determine if subplot is on the bottom row or leftmost column
        is_bottom = (idx // plot_shape[1]) == (plot_shape[0] - 1)
        is_leftmost = (idx % plot_shape[1]) == 0

        # Set titles for the x and y axes if specified and subplot is in the correct position
        if style_config.get('titles_x') and is_bottom:
            # TODO: FIX THIS
            ax.set_xlabel(style_config['titles_x'])
        if style_config.get('titles_y') and is_leftmost:
            ax.set_ylabel(style_config['titles_y'])

        # Hide axis if specified in style_config
        if not style_config['axis_marker']:
            ax.axis('off')

    # If there are more subplots than images, clear the unused subplots
    for empty_ax in axes[len(images):]:
        empty_ax.axis('off')

    if save_path:
        # Validate save_type
        if save_type not in ["png", "svg"]:
            raise ValueError("Unsupported save_type. Expected 'png' or 'svg'.")

        if save_path.endswith(".png") or save_path.endswith(".svg"):
            save_path = save_path[:-4]

        # Save the figure to the specified path with the specified file type
        plt.savefig(f"{save_path}.{save_type}", format=save_type)
        plt.close()  # Close the plot explicitly after saving to avoid displaying it
    else:
        # If no save_path is provided, display the figure
        plt.show()


def _determine_plot_shape(num_images: int) -> Tuple[int, int]:
    """
    Determines an optimal plot shape (rows and columns) for displaying a given number of images.

    If the number of images is a perfect square, the function will return equal numbers of rows and columns.
    For other numbers, it aims to find a balance between the rows and columns to make the grid as square as possible.

    Args:
        num_images (int): The total number of images to be displayed.

    Returns:
        Tuple[int, int]: A tuple containing the number of rows (len_y) and
            the number of columns (len_x) for the plot grid.
    """

    # Calculate the square root of the number of images to start determining the grid size
    root = math.sqrt(num_images)

    # Handle special cases
    if num_images == 1:
        return 1, 1
    elif num_images <= 3:
        return 1, num_images
    elif int(root + 0.5) ** 2 == num_images:
        sq_num = int(root)
        return sq_num, sq_num
    else:
        # Initialize variables to store the best found factor pair
        best_diff = float('inf')
        best_pair = (1, num_images)

        # Iterate over possible divisors to find the closest pair
        for i in range(1, int(num_images ** 0.5) + 1):
            if num_images % i == 0:
                j = num_images // i
                # Update if the current pair is closer to a square than previous best
                if abs(i - j) < best_diff:
                    best_diff = abs(i - j)
                    best_pair = (i, j)

        # Ensure the larger number is always the number of columns for consistency
        len_y, len_x = sorted(best_pair)

        return len_y, len_x


def _determine_image_type(input_img: np.ndarray) -> Optional[str]:
    """
    Determines the type of image based on its shape and pixel value range.

    This function analyzes an input image to classify it as either grayscale,
    color, binary, probabilistic (prob), segmented, or undefined. The classification
    is based on the dimensionality of the image array and the range of its pixel values.

    Args:
        input_img (np.ndarray): The image to be analyzed, represented as a NumPy array.

    Returns:
        Optional[str]: A string indicating the type of the image. Possible return values
        include "binary", "gray", "color", "prob", "segmented", or "undefined". Returns
        None if the input image is empty or None.

    Raises:
        ValueError: If the input is not a valid image array (e.g., if it's not a NumPy array).
    """

    # Validate input
    if input_img is None or not isinstance(input_img, np.ndarray):
        raise ValueError("Input must be a non-empty numpy.ndarray")

    if not input_img.size or input_img.ndim not in [2, 3]:
        return None  # Not a valid image

    # Check for binary and probabilistic images
    if input_img.ndim == 2:
        if np.nanmax(input_img) <= 1:
            if np.array_equal(input_img, input_img.astype(bool)):
                return "binary"
            else:
                return "prob"
        elif np.nanmax(input_img) > 8:
            return "gray"
        else:
            return "segmented"

    # Check for color images
    elif input_img.ndim == 3:
        if input_img.shape[0] == 3 and not np.any(input_img.shape == 3):
            input_img = np.moveaxis(input_img, 0, -1)
        if input_img.shape[-1] == 3:
            return "color"
        else:
            return "undefined"

    return "undefined"


def _create_segmented_cmap():
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


def _normalize_color(color):
    """Normalize color or colors from 0-255 range to 0-1 range."""
    if isinstance(color, str):
        return mcolors.to_rgba(color)
    if isinstance(color, tuple):  # Single color tuple
        return tuple(c / 255.0 for c in color)
    elif isinstance(color, list):  # List of color tuples
        return [tuple(c / 255.0 for c in col) for col in color]
    else:
        raise ValueError("Color must be a tuple or a list of tuples.")
