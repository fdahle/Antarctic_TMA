import geopandas as gpd

def load_shape_data(path_to_file: str, verbose: bool = False) -> gpd.GeoDataFrame:
    """
    Load shape data from a specified file path and return it as a GeoDataFrame.

    Args:
        path_to_file: The path to the shapefile.
        verbose: If true, prints information about the execution of the function.

    Returns:
        A GeoDataFrame containing the data from the shapefile.
    """

    data = gpd.read_file(path_to_file)
    
    if verbose:
        p.print_v(f"Shape data from {path_to_file} successfully loaded")

    return data
