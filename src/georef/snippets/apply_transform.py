import numpy as np
import rasterio

def apply_transform(image, transform, save_path, epsg_code=3031):

    if type(transform) == np.ndarray:
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