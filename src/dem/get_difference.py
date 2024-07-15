import numpy as np
from rasterio.warp import reproject, Resampling

def get_difference(dem1, dem2,
                   transform1=None, transform2=None,
                   no_data_val=-9999, epsg_code=3031):

    if transform1 == transform2:

        if dem1.shape == dem2.shape:
            # Direct calculation of difference possible
            difference = dem1 - dem2
        else:
            raise ValueError("DEM shapes do not match")
    else:

        # Prepare output array
        dem2_reprojected = np.empty_like(dem1, dtype=np.float32)

        # Reproject dem2 to match dem1
        reproject(
            source=dem2,
            destination=dem2_reprojected,
            src_transform=transform2,
            src_crs={'init': f'epsg:{epsg_code}'},
            dst_transform=transform1,
            dst_crs={'init': f'epsg:{epsg_code}'},
            resampling=Resampling.bilinear,
            src_nodata=no_data_val,
            dst_nodata=no_data_val
        )

        # Calculate the difference and handle no data values
        difference = np.abs(dem1 - dem2_reprojected)

        # Set no data values to np.nan
        difference[dem1 == no_data_val] = np.nan
        difference[dem2_reprojected == no_data_val] = np.nan

    return difference
