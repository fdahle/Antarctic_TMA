import copy

import geopandas as gpd
import matplotlib.pyplot as plt
import pyproj

from shapely.ops import transform

def display_shapes(input_shapes, normalize=False, title=None, save_path=None):

    shape_data = copy.deepcopy(input_shapes)

    fig, ax = plt.subplots(figsize=(10, 10))

    print(shape_data)

    if normalize:
        shape_data = _normalize_geometries(shape_data)

    print(shape_data)


    # Plot the GeoDataFrame
    shape_data.plot(ax=ax, alpha=0.5)  # Adjust alpha to change the transparency, makes it easier to visualize overlaps

    plt.axis('equal')  # Set equal scaling by changing axis limits
    plt.show()

def _normalize_geometries(gdf):
    """
    Normalizes the geometries in a GeoDataFrame so that the minimum x and y are 0.
    """
    minx, miny, maxx, maxy = gdf.total_bounds

    print(minx, miny, maxx, maxy)


    # Apply normalization (translation) to each geometry
    gdf.geometry = gdf.geometry.apply(lambda geom: _translate_geom(geom, minx, miny))
    return gdf

def _translate_geom(geom, xoff, yoff):
    transformer = pyproj.Transformer.from_proj(
        pyproj.Proj(proj='affine', s11=1, s12=0, s21=0, s22=1, xoff=-xoff, yoff=-yoff),
        pyproj.Proj(proj='affine', s11=1, s12=0, s21=0, s22=1, xoff=0, yoff=0),
        always_xy=True)
    return transform(transformer.transform, geom)
