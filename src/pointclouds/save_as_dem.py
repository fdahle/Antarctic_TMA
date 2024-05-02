# Package imports
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata


def save_as_dem(points_arr, output_path, grid_size=10, method='linear', epsg=3031):

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
