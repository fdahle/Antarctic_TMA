import matplotlib.colors

"""
This function creates a colormap for the segmentation
"""

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

    custom_norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
    custom_cmap = matplotlib.colors.ListedColormap(colors)

    return custom_cmap, custom_norm
