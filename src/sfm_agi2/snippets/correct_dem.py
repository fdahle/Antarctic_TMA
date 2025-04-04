import numpy as np

def correct_dem(dem_old, dem_new,
                mask_old=None, mask_new=None):

    # make sure the dems have the same shape
    if dem_old.shape != dem_new.shape:
        raise ValueError("Old and new DEM have different shapes")

    # make sure mask is the same shape as the dems
    if mask_old is not None:
        if mask_old.shape != dem_old.shape:
            raise ValueError("The mask has a different shape than the DEMs")
    else:
        mask_old = np.ones(dem_old.shape, dtype=bool)
    if mask_new is not None:
        if mask_new.shape != dem_new.shape:
            raise ValueError("The mask has a different shape than the DEMs")
    else:
        mask_new = np.ones(dem_new.shape, dtype=bool)

    # calculate difference between the dems
    diff = dem_new - dem_old

