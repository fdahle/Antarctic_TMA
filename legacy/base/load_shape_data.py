import geopandas as gpd

import base.print_v as p


def load_shape_data(path_to_file, catch=True, verbose=False, pbar=None):
    """
    load_shape_data(path_to_file, catch, verbose, pbar):
    Load the shape data from a file and return the data as a geopandas array.
    Args:
        path_to_file (String): The path to the shapefile
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar
    Returns:
        data (GeoPandas): The data of the shapefile
    """

    p.print_v("Start: load_shape_data", verbose, pbar=pbar)

    try:
        data = gpd.read_file(path_to_file)
    except (Exception,) as e:
        if catch:
            p.print_v(f"Loading shape data from '{path_to_file}' failed",
                      verbose, "red", pbar=pbar)
            return None
        else:
            raise e

    p.print_v(f"Shape data from {path_to_file} successfully loaded",
              verbose, pbar=pbar)

    return data
