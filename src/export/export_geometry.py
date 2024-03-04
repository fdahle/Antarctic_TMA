import os
import pandas as pd

from typing import Union, Optional

import geopandas as gpd
from shapely.geometry import shape
from shapely.wkt import loads


def export_geometry(geometry: Union[str, shape],
                    output_path: str,
                    attributes: pd.DataFrame = None,
                    key_field: Optional[str] = None,
                    crs: Union[int, str] = 'EPSG:3031',
                    attach: bool = False,
                    overwrite_file: bool = False,
                    overwrite_entry: bool = False) -> None:
    """
    Exports a given Shapely geometry (or equivalent WKT string) to a GeoDataFrame and saves it to
    a file in either SHP or GeoJSON format. The function allows for appending to, overwriting an
    entire file, or updating an existing entry in the file.

    Args:
        geometry (Union[str, shape]): A Shapely geometry object or its WKT string representation.
        output_path (str): The file path where the geometry should be saved.
        attributes (pd.DataFrame): Attributes as a dataframe that can be saved alongside the
            geometry. Defaults to None.
        key_field (Optional[str], optional): The unique attribute key to identify existing entries
            for updating. Required if 'overwrite_entry' is True.
        crs (Union[int, str], optional): The coordinate reference system to use.
            Defaults to 'EPSG:3031'.
        attach (bool, optional): If True, appends the geometry to an existing file.
            Defaults to False.
        overwrite_file (bool, optional): If True, overwrites the entire existing file.
            Defaults to False.
        overwrite_entry (bool, optional): If True, updates an existing entry based on 'key_field'.
            Defaults to False.
    Raises:
        ValueError: For various conditions such as
            - conflicting parameters,
            - unsupported file extensions,
            - missing 'key_field' when required
    """

    # raise error for conflicting overwrite parameters
    if overwrite_file and overwrite_entry:
        raise ValueError("'overwrite_file' and 'overwrite_entry' cannot both be true.")
    if overwrite_file and attach:
        raise ValueError("'overwrite_file' and 'attach' cannot both be true.")
    if overwrite_entry and not attach:
        raise ValueError("'overwrite_entry' requires 'attach' to be True.")

    # raise error for missing key field
    if overwrite_entry and not key_field:
        raise ValueError("'key_field' must be specified when 'overwrite_entry' is True.")

    # raise error for missing key field in attributes
    if overwrite_entry and key_field not in attributes:
        raise ValueError(f"'key_field' '{key_field}' must be present in attributes when "
                         f"'overwrite_entry' is used.")

    # raise an error for file conflict
    if os.path.isfile(output_path) and not overwrite_file and not overwrite_entry and \
            not attach:
        raise FileExistsError(f"'{output_path}' already exists.")

    # raise error for file extension
    file_extension = os.path.splitext(output_path)[1].lower()
    if file_extension not in ['.shp', '.geojson']:
        raise ValueError(f"File extension '{file_extension}' is not supported.")

    # Convert WKT string to Shapely geometry if necessary
    if isinstance(geometry, str):
        geometry = loads(geometry)

    # Prepare the data for GeoDataFrame
    data = {'geometry': [geometry]}
    if attributes is not None:
        for key, value in attributes.items():
            if isinstance(value, pd.Series):
                if not value.empty:
                    value = value.iloc[0]  # Assuming you want the first item if it's a Series
                else:
                    raise ValueError(f"The attribute '{key}' is an empty Series.")
            data[key] = [value]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(data, crs=crs)

    # Easy case: no file existing, or we just overwrite file
    if os.path.isfile(output_path) is False or overwrite_file:
        gdf.to_file(output_path, driver='GeoJSON' if file_extension == '.geojson' else None)
        return

    # attach data to the existing file
    if attach:

        # get existing data
        existing_gdf = gpd.read_file(output_path)

        # raise error if key field is missing
        if key_field not in existing_gdf:
            raise ValueError(f"Key field '{key_field}' not found in the file.")

        # check if key field value already exists
        if any(attr in existing_gdf[key_field].values for attr in attributes[key_field]):

            # remove existing entry if overwrite is True
            if overwrite_entry:

                existing_gdf = existing_gdf[existing_gdf[key_field] != attributes[key_field].iloc[0]]

            # raise error because we are not allowed to overwrite
            else:
                raise ValueError(f"Key field {key_field} ({existing_gdf[key_field]}) already exists "
                                 f"in the file. Use 'overwrite_entry' to update.")

        # append the new data
        gdf = pd.concat([existing_gdf, gdf], ignore_index=True)

        # save to file
        gdf.to_file(output_path, driver='GeoJSON' if file_extension == '.geojson' else None)
