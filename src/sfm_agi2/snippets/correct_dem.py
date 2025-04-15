import numpy as np
import xdem

from rasterio.transform import Affine


def correct_dem(dem_old: np.ndarray, dem_new: np.ndarray,
                transform_old: Affine, transform_new: Affine,
                mask_old: np.ndarray | None = None,
                mask_new: np.ndarray | None = None,
                slope_new: np.ndarray | None = None,
                crs: int = 3031, no_data_val: int | float = np.nan):

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

    # convert the dems to xdem.DEM
    dem_old = xdem.DEM.from_array(dem_old, transform=transform_old,
                                    crs=3031, nodata=np.nan)
    dem_new = xdem.DEM.from_array(dem_new, transform=transform_new,
                                    crs=3031, nodata=np.nan)

    # calc slope if not provided
    if slope_new is None:
        slope_new = xdem.terrain.slope(dem_new)

    # combine both masks
    mask = np.logical_and(mask_old, mask_new)

    # init the classes for dem correction
    deramp = xdem.coreg.Deramp()
    nuth_kaab = xdem.coreg.NuthKaab()

    # fit the deramp model and apply the model
    deramp.fit(dem_new, dem_old, mask=mask)
    dem_aligned = deramp.apply(dem_old)

    # fit co-registration model and apply the model
    nuth_kaab.fit(dem_new, dem_aligned, mask=mask)
    dem_aligned = nuth_kaab.apply(dem_aligned)

    # get the data from the aligned dem
    dem_aligned = dem_aligned.data

    # get the new matrix from nuth kaab
    matrix_aligned = nuth_kaab.to_matrix()

    # update the transform
    transform_aligned = Affine(
        transform_old.a, transform_old.b,
        transform_old.c + matrix_aligned[0, 3],
        transform_old.d, transform_old.e,
        transform_old.f + matrix_aligned[1, 3]
    )

    return dem_aligned, transform_aligned