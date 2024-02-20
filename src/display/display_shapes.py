import copy

import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj

from shapely.ops import transform, unary_union
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads as wkt_loads

base_style_config = {
    "axis_marker": True,
    "title": None,
    "labels": None,
}

def display_shapes(input_shapes, normalize=False, title=None, save_path=None):

    shapes = copy.deepcopy(input_shapes)

    # convert all input data to geo-series but also save the geoms for normalizing
    geoms = []
    for i, shape in enumerate(shapes):

        # Convert WKT strings to Shapely geometry first
        if isinstance(shape, str):
            shape = wkt_loads(shape)

        # Convert elements to plottable geo-series
        if isinstance(shape, BaseGeometry):
            geoms.append(shape)
            shapes[i] = gpd.GeoSeries([shape])
        elif isinstance(shape, list):
            shapes[i] = gpd.GeoSeries(shape)
            for item in shape:
                geoms.append(item)

    # find global min_x and min_y
    all_geoms = unary_union(geoms)
    min_x, min_y, _, _ = all_geoms.bounds

    # Normalize the geometries
    if normalize:
        shapes = _normalize_geometries(shapes, min_x, min_y)

    # create figure on which the shapes will be plotted
    fig, ax = plt.subplots(figsize=(10, 10))

    # plot the shapes
    for shape in shapes:
        shape.plot(ax=ax, alpha=0.5)

    # the actual plotting
    plt.axis('equal')  # Set equal scaling by changing axis limits
    plt.show()


def _normalize_geometries(shapes, min_x, min_y):

    normalized_shapes = []
    for gseries in shapes:
        # Apply translation to each geometry in the GeoSeries
        translated_geoms = gseries.geometry.apply(lambda geom: _translate_geom(geom, xoff=-min_x, yoff=-min_y))
        normalized_gseries = gpd.GeoSeries(translated_geoms)
        normalized_shapes.append(normalized_gseries)

    return normalized_shapes


def _translate_geom(geom, xoff, yoff):
    """
    Translates a Shapely geometry by the given x and y offsets.
    Args:
        geom: Shapely geometry to translate
        xoff: x offset
        yoff: y offset

    Returns:
          Translated Shapely geometry
    """

    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj(proj='affine', s11=1, s12=0, s21=0, s22=1, xoff=-xoff, yoff=-yoff),
        pyproj.Proj(proj='affine', s11=1, s12=0, s21=0, s22=1, xoff=0, yoff=0),
        always_xy=True)
    return transform(transformer.transform, geom)
