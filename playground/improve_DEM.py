import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import match_template


def outlier_filter(diff, f=3, max_dz=100):
    print("Removing outliers")
    print("Initial pixel count:", np.count_nonzero(~np.isnan(diff)))
    # Absolute dz filter
    diff = np.where(np.abs(diff) > max_dz, np.nan, diff)
    print("After absolute dz filter:", np.count_nonzero(~np.isnan(diff)))

    # MAD filter to exclude extreme values
    mad = np.nanmedian(np.abs(diff - np.nanmedian(diff)))
    threshold = f * mad
    diff = np.where(np.abs(diff - np.nanmedian(diff)) > threshold, np.nan, diff)
    print("After MAD filter:", np.count_nonzero(~np.isnan(diff)))
    return diff


def get_filtered_slope(arr, slope_lim=(0.1, 40)):
    # Apply a Gaussian filter to estimate slopes
    gradient_x, gradient_y = np.gradient(arr)
    slope = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    slope = np.where((slope >= slope_lim[0]) & (slope <= slope_lim[1]), slope, np.nan)
    return slope


def compute_offset(ref_dem, src_dem, mode='nuth', max_dz=100, slope_lim=(0.1, 40)):
    diff = src_dem - ref_dem
    print("Initial elevation difference stats:")
    print("Mean:", np.nanmean(diff), "Median:", np.nanmedian(diff))

    # Remove outliers
    diff = outlier_filter(diff, max_dz=max_dz)

    # Slope filtering
    slope = get_filtered_slope(ref_dem, slope_lim=slope_lim)
    diff = np.where(~np.isnan(slope), diff, np.nan)

    print("Filtered elevation difference stats:")
    print("Mean:", np.nanmean(diff), "Median:", np.nanmedian(diff))

    # Compute offset based on specified mode
    dx, dy = 0, 0
    if mode == 'ncc':
        # Use normalized cross-correlation for alignment
        result = match_template(ref_dem, src_dem, pad_input=True)
        y_offset, x_offset = np.unravel_index(np.argmax(result), result.shape)
        dy, dx = y_offset - ref_dem.shape[0] // 2, x_offset - ref_dem.shape[1] // 2
    elif mode == 'nuth':
        # Calculate offset using slope and aspect
        aspect = np.arctan2(gradient_y, gradient_x)
        fit_param = np.nanmedian(diff) / np.tan(np.nanmedian(slope))
        dx = fit_param * np.sin(aspect)
        dy = fit_param * np.cos(aspect)

    return dx, dy, diff
