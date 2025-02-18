import numpy as np
import xdem


def create_slope_mask(dem, transform, mask, max_slope):
    # pass the dem to xdem
    xdem_modern = xdem.DEM.from_array(dem, transform, crs=3031)

    # get slope
    slope = xdem.terrain.slope(xdem_modern)

    # create initial slope mask
    slope_mask = slope < max_slope
    slope_mask = slope_mask.data
    slope_mask = np.ma.getdata(slope_mask)

    # adapt mask based on confidence
    mask[slope_mask == 0] = 0

    return mask