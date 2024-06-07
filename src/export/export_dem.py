# Package imports
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata


def export_dem(points_arr: np.ndarray, output_path: str, grid_size: float = 10,
               method: str = 'linear', epsg: int = 3031) -> None:
    """
    Create a DEM from a point cloud and export it as a GeoTIFF file. The point cloud is converted
    to a regular grid with a specified grid size. The z-values at grid points are determined using
    interpolation.
    Args:
        points_arr (np.ndarray): A NumPy array with shape (n, 3) where each row represents
                                 x, y, and z coordinates of a point.
        output_path (str): Path where the GeoTIFF file will be saved.
        grid_size (float, optional): The grid size in the units of the coordinate system.
                                     Defaults to 10.
        method (str, optional): Method of interpolation used to determine the z-values
                                at grid points. Can be 'linear', 'nearest', 'cubic', etc.
                                Defaults to 'linear'.
        epsg (int, optional): The EPSG code that represents the coordinate reference system
                              of the output DEM. Defaults to 3031 (Antarctic Polar Stereographic).
    Returns:

    """

    # get x, y, z values
    x = points_arr[:, 0]
    y = points_arr[:, 1]
    z = points_arr[:, 2]

    # create a grid
    xi = np.arange(min(x), max(x), grid_size)
    yi = np.arange(min(y), max(y), grid_size)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate z values on the grid
    zi = griddata((x, y), z, (xi, yi), method=method)

    # Create a new raster file
    transform = from_origin(min(x), max(y), grid_size, grid_size)
    new_dataset = rasterio.open(
        output_path, 'w', driver='GTiff',
        height=zi.shape[0], width=zi.shape[1],
        count=1, dtype=str(zi.dtype),
        crs=f'EPSG:{epsg}',
        transform=transform
    )

    # Write the DEM data
    new_dataset.write(zi[::-1], 1)  # Invert Y axis to match the raster format
    new_dataset.close()
