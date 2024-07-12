import numpy as np

import src.display.display_images as di


def register_dem(dem, footprints,
                 ortho=None,
                 no_data_value=-99999):

    dem[dem == no_data_value] = np.nan

    from playground.create_bounding_rectangle import create_bounding_rectangle as cbr
    bounding_rectangle = cbr(footprints)

    print(bounding_rectangle.bounds)

    # get dem for the bounding rectangle
    import src.load.load_rema as lr
    dem_rema = lr.load_rema(bounding_rectangle, zoom_level=32)

    # get the satellite image for the bounding rectangle if ortho is provided
    if ortho is not None:
        import src.load.load_satellite as ls
        sat, sat_transform = ls.load_satellite(bounding_rectangle)

    # Normalize the DEM
    dem = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem))

    # normalize the rema dem
    dem_rema = (dem_rema - np.nanmin(dem_rema)) / (np.nanmax(dem_rema) - np.nanmin(dem_rema))

    # resize the input dem to the size of the rema dem
    dem_resized = resize_dem_with_nan(dem, dem_rema.shape, order=3)

    import src.dem.find_peaks_in_DEM as fpd
    peaks_resized = fpd.find_peaks_in_dem(dem_resized, no_data_value=-32767)
    peaks_rema = fpd.find_peaks_in_dem(dem_rema)

    # Convert peaks to keypoints
    keypoints_resized = np.array(peaks_resized)
    keypoints_rema = np.array(peaks_rema)

    # resize ortho to sat size
    if ortho is not None:
        from scipy.ndimage import zoom
        zoom_factors = (sat.shape[1] / ortho.shape[0], sat.shape[2] / ortho.shape[1])

        # Resize the array using the calculated zoom factors
        ortho_resized = zoom(ortho, zoom_factors, order=3)

    # find tps between ortho and sat
    import src.base.find_tie_points as ftp
    tpd = ftp.TiePointDetector('lightglue')
    points, conf = tpd.find_tie_points(sat, ortho_resized)

    import cv2
    _, filtered = cv2.findHomography(points[:, 0:2], points[:, 2:4], cv2.RANSAC, 5.0)
    filtered = filtered.flatten()

    # 1 means outlier
    points = points[filtered == 0]
    conf = conf[filtered == 0]

    print(f"{np.count_nonzero(filtered)} outliers removed with RANSAC")

    # adjust points for the resized ortho
    #points[:, 2] = points[:, 2] * (1 / zoom_factors[1])
    #points[:, 3] = points[:, 3] * (1 / zoom_factors[0])

    #di.display_images([sat, ortho_resized], tie_points=points, tie_points_conf=conf)

    print(sat_transform.shape)
    print(points[:, 0:2].shape)
    exit()

    # convert tie-points to absolute values
    absolute_points = np.array([sat_transform * tuple(point) for point in points[:, 0:2]])
    points[:, 0:2] = absolute_points

    import src.georef.snippets.calc_transform as ct
    transform, residuals = ct.calc_transform(dem_resized, points,
                                             transform_method="rasterio",
                                             gdal_order=3)

    import src.georef.snippets.apply_transform as at
    at.apply_transform(dem_resized, transform,
                       save_path="/home/fdahle/Desktop/agi_test/calc/dem_resized.tif")

    print("DEM", dem.shape, dem.shape[1]/dem.shape[0])
    print("DEM_REMA", dem_rema.shape, dem_rema.shape[1]/dem_rema.shape[0])
    print("DEM_RESIZED", dem_resized.shape, dem_resized.shape[1]/dem_resized.shape[0])
    print("SAT", sat.shape, sat.shape[2]/sat.shape[1])
    print("ORTHO", ortho.shape, ortho.shape[1]/ortho.shape[0])
    print("ORTHO_RESIZED", ortho_resized.shape, ortho_resized.shape[1]/ortho_resized.shape[0])
    exit()

    if ortho is None:
        di.display_images([dem_rema, dem_resized], points=[keypoints_rema, keypoints_resized])
    else:
        di.display_images([dem_rema, dem_resized, sat, ortho],
                          points=[keypoints_rema, keypoints_resized, [], []])



def resize_dem_with_nan(dem: np.ndarray, target_shape: tuple, order: int = 3) -> np.ndarray:
    """
    Resizes a DEM array containing NaN values.

    Args:
        dem (np.ndarray): 2D numpy array representing the DEM.
        target_shape (tuple): Desired shape (rows, cols) for the resized DEM.
        order (int, optional): The order of the spline interpolation used to resize. Default is 3.

    Returns:
        np.ndarray: Resized DEM with NaN values correctly handled.
    """

    from scipy.ndimage import zoom

    # Create a mask of NaNs
    nan_mask = np.isnan(dem)

    # Replace NaNs with a temporary value (e.g., the minimum value of the DEM)
    temp_value = np.nanmin(dem)
    dem_temp = np.where(nan_mask, temp_value, dem)

    # Calculate zoom factors
    zoom_factors = (target_shape[0] / dem.shape[0], target_shape[1] / dem.shape[1])

    # Resize the DEM with the temporary values
    dem_resized = zoom(dem_temp, zoom_factors, order=order)

    # Resize the NaN mask
    nan_mask_resized = zoom(nan_mask.astype(float), zoom_factors, order=0) > 0.5

    # Apply the resized NaN mask to the resized DEM
    dem_resized[nan_mask_resized] = np.nan

    return dem_resized



if __name__ == "__main__":

    _image_ids = ["CA184832V0146", "CA184832V0147", "CA184832V0148", "CA184832V0149", "CA184832V0150"]
    _no_data_value = -32767

    import src.load.load_image as li
    _dem_path = "/home/fdahle/Desktop/agi_test/output/dem_relative.tif"
    _dem = li.load_image(_dem_path)

    _ortho_path = "/home/fdahle/Desktop/agi_test/output/ortho_relative.tif"
    _ortho = li.load_image(_ortho_path)

    import src.base.connect_to_database as ctd
    from shapely import wkt

    conn = ctd.establish_connection()
    _image_ids_str = "','".join(_image_ids)
    sql_string = (f"SELECT st_astext(footprint_exact) FROM images_georef_3 "
                  f" WHERE image_id IN ('{_image_ids_str}')")
    data = ctd.execute_sql(sql_string, conn)
    footprints = data['st_astext'].tolist()
    footprints = [wkt.loads(footprint) for footprint in footprints]

    register_dem(_dem, footprints, ortho=_ortho, no_data_value=_no_data_value)