import numpy as np
import xdem

import src.display.display_images as di

debug_display_data = False

def create_slope_mask(dem, transform, existing_mask, max_slope):
    # pass the dem to xdem
    xdem_modern = xdem.DEM.from_array(dem, transform, crs=3031)

    # get slope
    slope = xdem.terrain.slope(xdem_modern)

    # create initial slope mask
    slope_mask = slope < max_slope
    slope_mask = slope_mask.data
    slope_mask = np.ma.getdata(slope_mask)

    # adapt mask based on confidence
    existing_mask[slope_mask == 0] = 0

    if debug_display_data:
        # convert slope to numpy array
        slope = slope.data
        di.display_images([slope, slope_mask, existing_mask], ["Slope", "Slope Mask", "Existing Mask"])

    return existing_mask