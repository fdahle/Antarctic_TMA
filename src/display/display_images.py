import math
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Dict, Any

base_style_config = {
    "axis_marker": True,
    "point_color": (255, 0, 0),
    "point_size": 7,
    "plot_shape": "auto",
    "line_color": (255, 0, 0),
    "line_width": 1,
    "title": None,
    "titles_sup": None,
    "titles_x": None,
    "titles_y": None
}

def display_images(images: List[Any],
                   points: Optional[List[List[Tuple[int, int]]]] = None,
                   lines: Optional[List[List[Tuple[int, int, int, int]]]] = None,
                   style_config: Optional[Dict[str, Any]] = None,
                   save_path: Optional[str] = None,
                   save_type: str = "png") -> None:

    # images must always be in a list
    if not isinstance(images, list):
        images = [images]

    # we need at least one image
    assert len(images) > 0, "No images to display."

    # If style_config is None, make it an empty dict to avoid TypeError when unpacking
    if style_config is None:
        style_config = {}

    # Merge the user's style_config with the base_style_config
    # The user's style_config will override any default settings if specified
    style_config = {**base_style_config, **style_config}

    print(style_config)

    # determine plot shape
    if style_config['plot_shape'] == "auto":
        plot_shape = _determine_plot_shape(len(images))
    else:
        plot_shape = style_config['plot_shape']

    # create the plot
    fig, axarr = plt.subplots(plot_shape[0], plot_shape[1], squeeze=False)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust space between plots

    # Set the figure's main title if specified
    if style_config['title']:
        fig.suptitle(style_config['title'])

    # Flatten axarr to simplify iterating over it if it's multidimensional
    axes = axarr.flatten()

    # Iterate over images and their corresponding axes
    for idx, img in enumerate(images):

        # get ax for that image
        ax = axes[idx]

        # show the image
        ax.imshow(img)

        # Optionally draw points on the image
        if points and idx < len(points):
            for point in points[idx]:
                ax.plot(point[0], point[1], 'o', color=style_config['point_color'],
                        markersize=style_config['point_size'])

        # Optionally draw lines on the image
        if lines and idx < len(lines):
            for line in lines[idx]:
                ax.plot([line[0], line[2]], [line[1], line[3]], color=style_config['line_color'],
                        linewidth=style_config['line_width'])

        # Set subtitles for each subplot if specified
        if style_config.get('titles_sup') and idx < len(style_config['titles_sup']):
            ax.set_title(style_config['titles_sup'][idx], fontsize=10)

        # Determine if subplot is on the bottom row or leftmost column
        is_bottom = (idx // plot_shape[1]) == (plot_shape[0] - 1)
        is_leftmost = (idx % plot_shape[1]) == 0

        # Set titles for the x and y axes if specified and subplot is in the correct position
        if style_config.get('titles_x') and is_bottom:
            ax.set_xlabel(style_config['titles_x'])
        if style_config.get('titles_y') and is_leftmost:
            ax.set_ylabel(style_config['titles_y'])

        # Hide axis if specified in style_config
        if not style_config['axis_marker']:
            ax.axis('off')

    # If there are more subplots than images, clear the unused subplots
    for empty_ax in axes[len(images):]:
        empty_ax.axis('off')

    plt.show()  # Display the plot


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
