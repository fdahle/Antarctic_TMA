# Package imports
import copy
import numpy as np
import rasterio


def apply_transform(image, transform, save_path, epsg_code=3031):
    # copy transform to avoid changing the original
    transform = copy.deepcopy(transform)

    if type(transform) == np.ndarray:

        # flatten transform if it is a 3x3 matrix
        if transform.shape[0] == 3:
            transform = transform.flatten()

        # remove last row if it is [0, 0, 1]
        if transform.shape[0] == 9 and np.allclose(transform[-3:], [0, 0, 1]):
            transform = transform[:-3]

        r_transform = rasterio.transform.Affine(*transform)

    else:
        r_transform = transform

    # define the CRS
    crs = f"EPSG:{epsg_code}"

    # Save the image as a GeoTIFF
    with rasterio.open(save_path, 'w', driver='GTiff',
                       height=image.shape[0], width=image.shape[1],
                       count=1, dtype=image.dtype,
                       crs=crs, transform=r_transform,
                       nodata=0
                       ) as dst:
        dst.write(image, 1)
