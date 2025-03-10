"""show shapes with matplotlib"""

import matplotlib
matplotlib.use('TkAgg')  # noqa: E402
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
    "colors": [],
    "lines": [],
    "alpha": []
}


def display_shapes(shapes: Union[List[BaseGeometry], List[str], gpd.GeoDataFrame],
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
            if style_config['labels'][i] is None:
                continue

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

    # check if the number of lines is equal to the number of shapes
    if style_config['lines'] and len(style_config['lines']) != len(shapes):
        raise ValueError("The number of lines must be equal to the number of shapes")

    # convert all input data to geo-series but also save the geoms for normalizing
    geoms = []
    gs_shapes = []  # shapes as geo-series
    if isinstance(shapes, gpd.GeoDataFrame):
        for geom in shapes.geometry:
            geoms.append(geom)
        gs_shapes.append(shapes.geometry)
    else:
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
            elif shape is None:
                pass
            else:
                raise ValueError("Invalid input shape type. Expected Shapely geometry, WKT string, or GeoDataFrame.")

    # get the bounds of all shapes
    min_x, min_y, max_x, max_y = unary_union(geoms).bounds

    # Normalize the geometries
    if normalize:
        gs_shapes = _normalize_geometries(gs_shapes, min_x, min_y)

        # flatten GeoSeries
        flat_geoms = [geom for geo_series in gs_shapes for geom in geo_series.geometry]

        # ensure all geoms are valid
        valid_geoms = [geom for geom in flat_geoms if geom.is_valid]

        # perform unary union
        union = unary_union(valid_geoms)

        # update bounds after normalization
        min_x, min_y, max_x, max_y = union.bounds

    # create figure on which the shapes will be plotted
    fig, ax = plt.subplots(figsize=(10, 10))

    # Limit plot to the bounds of the polygons
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Set the figure's main title if specified
    if style_config['title']:
        fig.suptitle(style_config['title'])

    # Hide axis if specified in style_config
    if not style_config['axis_marker']:
        ax.axis('off')

    if len(style_config['colors']) == 0:
        style_config['colors'] = ['gray'] * len(gs_shapes)

    # iterate all shapes
    for i, shape in enumerate(gs_shapes):

        # determine color
        face_color = style_config['colors'][i] if (
                    style_config['colors'] and i < len(style_config['colors']) and style_config['colors'][i] is not None) else "none"
        edge_color = style_config['lines'][i] if (
                    style_config['lines'] and i < len(style_config['lines'])) else face_color
        alpha = style_config['alpha'][i] if (
                    style_config['alpha'] and i < len(style_config['alpha'])) else 1

        print(face_color, edge_color, alpha)

        # plot the shape
        shape.plot(ax=ax, facecolor=face_color, edgecolor=edge_color, alpha=alpha)

        # add label to the shape
        if style_config['labels'] and style_config['labels'][i] is not None:

            # iterate all geometries from the shape
            for j, geom in enumerate(shape.geometry):
                centroid = geom.centroid
                plt.annotate(style_config['labels'][i][j], (centroid.x, centroid.y),
                             textcoords="offset points", xytext=(0, 10),
                             ha='center', fontsize=10)

    if save_path:
        # Validate save_type
        if save_type not in ["png", "svg"]:
            raise ValueError("Unsupported save_type. Expected 'png' or 'svg'.")

        # Save the figure to the specified path with the specified file type
        plt.savefig(f"{save_path}.{save_type}", format=save_type)
        plt.close()  # Close the plot explicitly after saving to avoid displaying it
    else:
        print("PLTSHOW")
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
    for geo_series in shapes:
        translated_geoms = geo_series.geometry.apply(lambda geom: _translate_geom(geom, x_off=-min_x, y_off=-min_y))
        normalized_geo_series = gpd.GeoSeries(translated_geoms)
        normalized_shapes.append(normalized_geo_series)

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
