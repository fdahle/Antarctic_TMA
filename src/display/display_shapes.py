import geopandas as gpd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from shapely.wkt import loads
from shapely.geometry.base import BaseGeometry

def display_shapes(shape_files: object, edge_color: object = 'black',
                   subtitles: object = None, colors: object = None, alphas: object = None,
                   title: object = None, save_path=None) -> object:
    """
    Display shapes from GeoDataFrames, GeoSeries, or lists containing WKT strings or Shapely geometries.

    Parameters:
        shape_files (list): A list containing GeoDataFrames, GeoSeries, or lists of WKT strings or Shapely geometries.
        edge_color (str, optional): Color of the edges for the plotted shapes. Default is 'black'.
        title (str, optional): Title of the plot. Default is None.

    Returns:
        None
    """

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    if alphas is None:
        alphas = [1] * len(shape_files)

    for i,elem in enumerate(shape_files):

        face_color = colors[i] if (colors is not None and i < len(colors)) else 'lightgray'

        if isinstance(elem, str):

            geom = loads(elem)
            geom = gpd.GeoSeries([geom])

            geom.plot(ax=ax, edgecolor=edge_color, facecolor=face_color, alpha=alphas[i])

            if subtitles is not None:
                centroid = geom.geometry.centroid
                plt.annotate(subtitles[i], (centroid.x, centroid.y), textcoords="offset points", xytext=(0, 10), ha='center')

        elif isinstance(elem, BaseGeometry):
            geom = gpd.GeoSeries([elem])

            try:
                geom.plot(ax=ax, edge_color=edge_color, facecolor=face_color, alpha=alphas[i])
            except:
                geom.plot(ax=ax, color=face_color, alpha=alphas[i])

            if subtitles is not None:
                centroid = geom.geometry.centroid
                plt.annotate(subtitles[i], (centroid.x, centroid.y), textcoords="offset points", xytext=(0, 10), ha='center')

        elif isinstance(elem, list):
            # Iterate through the list of geometries
            for j, geom in enumerate(elem):
                if isinstance(geom, str):
                    # Convert WKT string to Shapely geometry
                    geom = loads(geom)
                if not isinstance(geom, gpd.GeoSeries):
                    # Convert individual geometry to GeoSeries
                    geom = gpd.GeoSeries([geom])

                geom.plot(ax=ax, edgecolor=edge_color, facecolor=face_color, alpha=alphas[i])

                if subtitles is not None:
                    if subtitles[i] is not None:
                        centroid = geom.geometry.centroid
                        plt.annotate(subtitles[i][j], (centroid.x, centroid.y), textcoords="offset points", xytext=(0, 10), ha='center')

        elif isinstance(elem, (gpd.GeoDataFrame, gpd.GeoSeries, pd.Series)):

            # Iterate through the GeoDataFrame or GeoSeries
            for geom in elem:
                if isinstance(geom, str):
                    # Convert WKT string to Shapely geometry
                    geom = loads(geom)
                if not isinstance(geom, gpd.GeoSeries):
                    # Convert individual geometry to GeoSeries
                    geom = gpd.GeoSeries([geom])

                geom.plot(ax=ax, edgecolor=edge_color, facecolor=face_color, alpha=alphas[i])

                if subtitles is not None:
                    centroid = geom.geometry.centroid
                    plt.annotate(subtitles[i][j], (centroid.x, centroid.y), textcoords="offset points", xytext=(0, 10), ha='center')

        else:
            raise ValueError(f"{type(elem)} is still unsupported")

    # Set x and y axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    # Set the aspect ratio of x and y axes to be equal
    #ax.set_aspect('equal')

    # Set the title of the plot
    if title is not None:
        plt.title(title)

    # Display or save the plot
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)