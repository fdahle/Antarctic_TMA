import fiona
import geopandas as gpd
from shapely.geometry import shape
from shapely.validation import explain_validity

def load_shape_data(path_to_file: str,
                    verbose: bool = False) -> gpd.GeoDataFrame:
    """
    Load shape data from a specified file path and return it as a GeoDataFrame.
    :param path_to_file: The path to the shapefile.
    :param verbose: If true, prints information about the execution of the function.
    :return: A GeoDataFrame containing the data from the shapefile.
    """

    valid_features = []

    # Open with fiona to filter invalid geometries
    with fiona.open(path_to_file) as src:
        for feature in src:
            try:
                # Ensure that the geometry has enough coordinates to form a valid polygon
                geom = feature["geometry"]
                if geom is None:
                    if verbose:
                        print(f"Ignored feature with missing geometry")
                    continue

                coords = geom.get('coordinates', [])

                # Check if it's a polygon and has at least 4 coordinates (3 points + closure)
                if geom['type'] == 'Polygon':
                    if len(coords) == 0 or len(coords[0]) < 4:
                        if verbose:
                            print(f"Ignored invalid polygon with insufficient coordinates")
                        continue
                elif geom['type'] == 'MultiPolygon':
                    if len(coords) == 0 or len(coords[0][0]) < 4:
                        if verbose:
                            print(f"Ignored invalid multipolygon with insufficient coordinates")
                        continue

                # Convert to Shapely geometry
                shapely_geom = shape(geom)

                # Optionally check for validity
                if not shapely_geom.is_valid:
                    if verbose:
                        print(f"Ignored invalid geometry: {explain_validity(shapely_geom)}")
                    continue

                valid_features.append(feature)
            except Exception as e:
                if verbose:
                    print(f"Error processing feature: {e}")
                continue

    # Load valid features into a GeoDataFrame
    valid_gdf = gpd.GeoDataFrame.from_features(valid_features)

    if verbose:
        print(f"Shape data from {path_to_file} successfully loaded with {len(valid_gdf)} valid geometries")

    return valid_gdf
