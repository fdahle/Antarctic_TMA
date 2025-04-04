def estimate_dem_quality(dem, modern_dem):

    quality_dict: dict[str, float] = {}

    # Calculate statistics for the whole DEM
    quality_dict = _calc_stats("all", modern_dem, dem_abs, quality_dict)

    # If a mask is provided, apply it and recalculate statistics for the masked area.
    if mask is not None:
        modern_dem = modern_dem.copy()  # To avoid modifying the original array.
        modern_dem[mask == 0] = np.nan
        quality_dict = _calc_stats("mask", modern_dem, dem_abs, quality_dict)
