import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj

from shapely.ops import transform, unary_union
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads as wkt_loads
from typing import List, Union, Dict, Optional, Any

base_style_config = {
    "axis_marker": True,
    "title": None,
    "labels": [],
    "colors": []
}


def display_shapes(shapes: Union[List[BaseGeometry], List[str]],
                   normalize: bool = False,
                   style_config: Optional[Dict[str, Any]] = None,
                   save_path: Optional[str] = None,
                   save_type: str = "png") -> None:
    """
        Displays or saves a series of shapes with optional normalization and annotations.

    Args:
        shapes: A list of input shapes as Shapely geometries or WKT strings.
        normalize: If True, translates shapes to a common origin.
        style_config: A dictionary for customizing plot appearance (colors, labels, etc.).
        save_path: Optional path to save the plot image.
        save_type: The file type to use when saving the plot image. Options are "png" or "svg".
    """

    # If style_config is None, make it an empty dict to avoid TypeError when unpacking
    if style_config is None:
        style_config = {}

    # Merge the user's style_config with the base_style_config
    # The user's style_config will override any default settings if specified
    style_config = {**base_style_config, **style_config}

    # check if the number of labels is equal to the number of shapes
    if style_config['labels']:
        if len(style_config['labels']) != len(shapes):
            raise ValueError("The number of labels must be equal to the number of shapes")

        # some shapes are lists, so check if the number of labels in this list is equal
        for i, shape in enumerate(shapes):
            if isinstance(shape, list):
                if len(style_config['labels'][i]) != len(shape):
                    raise ValueError("The number of labels must be equal to the number of elements "
                                     f"in the the {i}-th list")
            else:
                # put label in list so that displaying later is more straightforward
                style_config['labels'][i] = [style_config['labels'][i]]

    # check if the number of colors is equal to the number of shapes
    if style_config['colors'] and len(style_config['colors']) != len(shapes):
        raise ValueError("The number of colors must be equal to the number of shapes")

    # convert all input data to geo-series but also save the geoms for normalizing
    geoms = []
    gs_shapes = []  # shapes as geo-series
    for i, shape in enumerate(shapes):

        # Convert WKT strings to Shapely geometry first
        if isinstance(shape, str):
            shape = wkt_loads(shape)

        # Convert elements to plottable geo-series
        if isinstance(shape, BaseGeometry):
            geoms.append(shape)
            gs_shapes.append(gpd.GeoSeries([shape]))  # noqa
        elif isinstance(shape, list):
            gs_shapes.append(gpd.GeoSeries(shape))  # noqa
            for item in shape:
                geoms.append(item)
        else:
            raise ValueError("Invalid input shape type. Expected Shapely geometry or WKT string.")

    # find global min_x and min_y
    all_geoms = unary_union(geoms)
    min_x, min_y, _, _ = all_geoms.bounds

    # Normalize the geometries
    if normalize:
        gs_shapes = _normalize_geometries(gs_shapes, min_x, min_y)

    # create figure on which the shapes will be plotted
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set the figure's main title if specified
    if style_config['title']:
        fig.suptitle(style_config['title'])

    # Hide axis if specified in style_config
    if not style_config['axis_marker']:
        ax.axis('off')

    # iterate all shapes
    for i, shape in enumerate(gs_shapes):

        # determine color
        face_color = style_config['colors'][i] if (
                    style_config['colors'] and i < len(style_config['colors'])) else 'lightgray'

        # plot the shape
        shape.plot(ax=ax, facecolor=face_color, alpha=0.5)

        # add label to the shape
        if style_config['labels'] and style_config['labels'][i] is not None:

            # iterate all geometries from the shape
            for j, geom in enumerate(shape.geometry):
                centroid = geom.centroid
                plt.annotate(style_config['labels'][i][j], (centroid.x, centroid.y),
                             textcoords="offset points", xytext=(0, 10),
                             ha='center', fontsize=10)

    # Set equal scaling by changing axis limits
    plt.axis('equal')

    if save_path:
        # Validate save_type
        if save_type not in ["png", "svg"]:
            raise ValueError("Unsupported save_type. Expected 'png' or 'svg'.")

        # Save the figure to the specified path with the specified file type
        plt.savefig(f"{save_path}.{save_type}", format=save_type)
        plt.close()  # Close the plot explicitly after saving to avoid displaying it
    else:
        # If no save_path is provided, display the figure
        plt.show()


def _normalize_geometries(shapes: List[gpd.GeoSeries], min_x: float, min_y: float) -> List[gpd.GeoSeries]:
    """
    Normalizes geometries by translating them to have a common origin based on the minimum X and Y.

    Args:
        shapes: A list of GeoSeries containing the shapes to normalize.
        min_x: The global minimum X coordinate.
        min_y: The global minimum Y coordinate.

    Returns:
        A list of normalized GeoSeries.
    """

    normalized_shapes = []

    # Apply translation to each geometry in the GeoSeries
    for gseries in shapes:
        translated_geoms = gseries.geometry.apply(lambda geom: _translate_geom(geom, x_off=-min_x, y_off=-min_y))
        normalized_gseries = gpd.GeoSeries(translated_geoms)
        normalized_shapes.append(normalized_gseries)

    return normalized_shapes


def _translate_geom(geom: BaseGeometry, x_off: float, y_off: float) -> BaseGeometry:
    """
    Translates a Shapely geometry by the given X and Y offsets.
    Args:
        geom: The geometry to translate.
        x_off: X offset to add/remove.
        y_off: Y offset to add/remove.

    Returns:
        The translated geometry.
    """

    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj(proj='affine', s11=1, s12=0, s21=0, s22=1, xoff=-x_off, yoff=-y_off),
        pyproj.Proj(proj='affine', s11=1, s12=0, s21=0, s22=1, xoff=0, yoff=0),
        always_xy=True)
    return transform(transformer.transform, geom)
