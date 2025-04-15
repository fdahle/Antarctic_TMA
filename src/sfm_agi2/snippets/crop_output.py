import numpy as np
from affine import Affine

import src.base.calc_bounds as cb
import src.load.load_image as li
import src.export.export_tiff as et


def crop_output(pth_dem, pth_ortho,
                empty_pixels_dem, empty_pixels_ortho,
                use_lzw=False,
                crs=None,
                crop_mode="smallest"):

    print("[INFO] Crop DEM and ortho by removing empty pixels")

    # load the data
    dem, dem_transform = li.load_image(pth_dem, return_transform=True)
    ortho, ortho_transform = li.load_image(pth_ortho, return_transform=True)

    # Find valid content bounds
    dem_top, dem_bottom, dem_left, dem_right = _find_valid_crop_mask(dem, empty_pixels_dem)
    ortho_top, ortho_bottom, ortho_left, ortho_right = _find_valid_crop_mask(ortho, empty_pixels_ortho)

    if crop_mode == "smallest":
        top = max(dem_top, ortho_top)
        bottom = min(dem_bottom, ortho_bottom)
        left = max(dem_left, ortho_left)
        right = min(dem_right, ortho_right)
    elif crop_mode == "largest":
        top = min(dem_top, ortho_top)
        bottom = max(dem_bottom, ortho_bottom)
        left = min(dem_left, ortho_left)
        right = max(dem_right, ortho_right)
    else:
        raise ValueError("crop_mode must be 'smallest' or 'largest'")

    # Crop both arrays
    if dem.ndim == 3:
        cropped_dem = dem[:, top:bottom, left:right]
    else:
        cropped_dem = dem[top:bottom, left:right]

    if ortho.ndim == 3:
        cropped_ortho = ortho[:, top:bottom, left:right]
    else:
        cropped_ortho = ortho[top:bottom, left:right]

    # Update transform
    new_transform = dem_transform * Affine.translation(left, top)

    # calc new bounds
    new_bounds = cb.calc_bounds(new_transform, cropped_dem.shape)

    # save the data again
    et.export_tiff(cropped_dem, pth_dem, transform=new_transform,
                   use_lzw=use_lzw, crs=crs, no_data=empty_pixels_dem,
                   overwrite=True)
    et.export_tiff(cropped_ortho, pth_ortho, transform=new_transform,
                   use_lzw=use_lzw, crs=crs, no_data=empty_pixels_ortho,
                   overwrite=True)

    return new_bounds

def _find_valid_crop_mask(arr, nodata):
    """Return the indices of the first and last valid row and column."""
    mask = (arr == nodata)
    if arr.ndim == 3:
        # For multi-band, a pixel is nodata if all bands are nodata
        mask = np.all(mask, axis=0)

    valid_rows = np.any(~mask, axis=1)
    valid_cols = np.any(~mask, axis=0)

    top = np.argmax(valid_rows)
    bottom = len(valid_rows) - np.argmax(valid_rows[::-1])
    left = np.argmax(valid_cols)
    right = len(valid_cols) - np.argmax(valid_cols[::-1])

    return top, bottom, left, right