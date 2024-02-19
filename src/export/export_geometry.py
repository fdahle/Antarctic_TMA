import os
from typing import Union, Optional, Dict

import geopandas as gpd
from shapely.geometry import shape
from shapely.wkt import loads


def export_geometry(geometry: Union[str, shape],
                    output_path: str,
                    attributes: Optional[Dict[str, Union[str, int, float]]] = None,
                    key_attribute: Optional[str] = None,
                    crs: Union[int, str] = 'EPSG:3031',
                    attach: bool = False,
                    overwrite: bool = False) -> None:
    """
    Exports a given Shapely geometry (or equivalent WKT string) to a GeoDataFrame and saves it to
    a file in either SHP or GeoJSON format. The function also allows appending to or
    overwrite an existing file.

    Args:
        geometry (Union[str, shape]): A Shapely geometry object or its WKT string representation.
        output_path (str): The file path where the geometry should be saved.
        attributes (Optional[Dict[str, Union[str, int, float]]], optional): Attributes to be saved
            alongside the geometry. Defaults to None.
        crs (Union[int, str], optional): The coordinate reference system to use.
            Defaults to 'EPSG:3031'.
        attach (bool, optional): If True, appends the geometry to an existing file. Defaults to False.
        overwrite (bool, optional): If True, overwrites an existing file. Defaults to False.

    Raises:
        ValueError: If both 'attach' and 'overwrite' are set to True, or if the
            file extension is unsupported.
    """

    # TODO: Implement attach and overwrite true at the same time to overvrite the image_id

    if attach and overwrite:
        raise ValueError("'attach' and 'overwrite' cannot be true at the same time.")

    if os.path.isfile(output_path) and not attach and not overwrite:
        raise ValueError(f"'{output_path}' already exists.")

    file_extension = os.path.splitext(output_path)[1].lower()
    if file_extension not in ['.shp', '.geojson']:
        raise ValueError(f"File extension '{file_extension}' is not supported.")

    # Convert WKT string to Shapely geometry if necessary
    if isinstance(geometry, str):
        geometry = loads(geometry)

    # Prepare the data for GeoDataFrame
    data = {'geometry': [geometry]}
    if attributes:
        for key, value in attributes.items():
            data[key] = [value]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(data, crs=crs)

    # Exporting logic
    if attach and os.path.isfile(output_path):
        existing_gdf = gpd.read_file(output_path)
        gdf = existing_gdf.append(gdf, ignore_index=True)

    if overwrite or not os.path.isfile(output_path) or attach:
        if file_extension == '.shp':
            gdf.to_file(output_path)
        elif file_extension == '.geojson':
            gdf.to_file(output_path, driver='GeoJSON')
