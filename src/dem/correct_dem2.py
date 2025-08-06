# Library imports
import numpy as np
import xdem

# Local imports
import src.base.resize_image as ri


def correct_dem(historic_dem, modern_dem, hist_transform, modern_transform,
                historic_ortho=None, modern_ortho=None,
                mask=None, adapt_mask=False,
                max_slope=None, no_data_val=-9999, print_values=True):

    # check if the DEMs have the same shape
    if historic_dem.shape != modern_dem.shape:
        raise ValueError("The two DEMs have different shapes")

    # update mask with no data values
    if mask is not None:
        mask[historic_dem == no_data_val] = 0

    input_historic_dem = historic_dem.copy()

    # set no data values to nan
    modern_dem[historic_dem == no_data_val] = np.nan
    historic_dem[historic_dem == no_data_val] = np.nan

    # get mask as numpy array and resize to modern dem shape
    if mask is not None:
        mask = ri.resize_image(mask, modern_dem.shape)

    # make sure that mask is similar to the dems
    if mask is not None:
        if mask.shape != modern_dem.shape:
            raise ValueError("The mask has a different shape than the DEMs")


    # convert the dems to xdem.DEM
    historic_dem = xdem.DEM.from_array(historic_dem, transform=hist_transform,
                                       crs=3031, nodata=np.nan)
    modern_dem = xdem.DEM.from_array(modern_dem, transform=modern_transform,
                                     crs=3031, nodata=np.nan)

    # calculate the difference between the two DEMs
    print("Calc diff before")
    diff_before = modern_dem - historic_dem

    import src.display.display_images as di
    #di.display_images([historic_dem.data, modern_dem.data, diff_before.data])

    # calculate the slope of the modern DEM
    print("Calculating slope")
    slope = xdem.terrain.slope(modern_dem)

    # init the deramp & Nuth Kaab classes
    deramp = xdem.coreg.Deramp()
    nuth_kaab = xdem.coreg.NuthKaab()

    while True:
        try:
            # create a slope mask based on max_slope threshold
            if max_slope is None:
                slope_mask = np.ones_like(modern_dem.data, dtype=bool)
            else:
                slope_mask = slope < max_slope
                slope_mask = slope_mask.data
                slope_mask = np.ma.getdata(slope_mask)

            # create a mask for the stable ground
            if mask is not None:
                inlier_mask = mask & slope_mask
            else:
                inlier_mask = slope_mask

            # convert the mask to a boolean array
            inlier_mask = inlier_mask.astype(bool)

            if adapt_mask:
                # get modern dem as numpy array
                modern_data = modern_dem.data

                # extract inlier heights
                valid_heights = modern_data[inlier_mask]

                # get median height
                threshold_height = np.nanmedian(valid_heights)
                print("Threshold height", threshold_height)

                # update mask to only keep values above threshold
                inlier_mask &= (modern_data > threshold_height)

            #import src.display.display_images as di
            #di.display_images([historic_dem.data, modern_dem.data],
            #                  overlays=[inlier_mask, inlier_mask])

            # find coords of your inliers
            ys, xs = np.where(inlier_mask)
            minx, maxx = xs.min(), xs.max()
            miny, maxy = ys.min(), ys.max()

            # crop arrays & transforms
            h_crop = historic_dem.data[miny:maxy, minx:maxx]
            m_crop = modern_dem.data[miny:maxy, minx:maxx]
            t_crop = hist_transform * Affine.translation(minx, miny)

            # wrap into xdem.DEM
            h_dem_crop = xdem.DEM.from_array(h_crop, t_crop, crs=historic_dem.crs)
            m_dem_crop = xdem.DEM.from_array(m_crop, modern_dem.transform, crs=modern_dem.crs)
            h_dem_crop.nodata = historic_dem.nodata
            m_dem_crop.nodata = modern_dem.nodata

            deramp.fit(m_dem_crop, h_dem_crop, np.ones_like(h_crop, bool))
            aligned_crop = deramp.apply(h_dem_crop)
            nuth_kaab.fit(m_dem_crop, aligned_crop, np.ones_like(h_crop, bool))

            # apply the co-registration
            print("Apply Nuth Kääb")
            aligned_dem = deramp.apply(historic_dem)
            aligned_dem = nuth_kaab.apply(aligned_dem)
            aligned_matrix = nuth_kaab.to_matrix()  # noqa

            # get the difference after the co-registration
            print("Calc diff")
            diff_after = modern_dem - aligned_dem

            break

        except ValueError as e:

            if "Less than 10 different cells exist" in str(e):
                max_slope = max_slope + 5
                print(f"Coreg failed. Max slope increased to {max_slope}")
            else:
                raise e
            if max_slope == 90:
                raise ValueError("Max slope reached 90. Coreg failed.")
    """
    import matplotlib.pyplot as plt

    # get 75th percentile of the difference
    vmin_before = np.percentile(diff_before.data.compressed(), 20)
    vmax_before = np.percentile(diff_before.data.compressed(), 80)
    vmin_after = np.percentile(diff_after.data.compressed(), 20)
    vmax_after = np.percentile(diff_after.data.compressed(), 80)

    # get absolute v_min and v_max
    vmin = np.min([vmin_before, vmin_after])
    vmax = np.max([vmax_before, vmax_after])
    
    alpha = 0.2
    # Create a figure and two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Create a figure and two subplots
    im1 = axes[0].imshow(diff_before.data, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
    #axes[0].imshow(inlier_mask, cmap="Greys", alpha=alpha)  # Overlay the mask
    axes[0].set_title("Difference Before")
    plt.colorbar(im1, ax=axes[0], label="Elevation change (m)")

    # Plot diff_after on the second subplot
    im2 = axes[1].imshow(diff_after.data, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
    #axes[1].imshow(inlier_mask, cmap="Greys", alpha=alpha)  # Overlay the mask
    axes[1].set_title("Difference After")
    plt.colorbar(im2, ax=axes[1], label="Elevation change (m)")

    # put first axis of modern ortho in the last
    #modern_ortho = np.moveaxis(modern_ortho, 0, 2)
    #im3 = axes[2].imshow(modern_ortho)
    #axes[2].imshow(inlier_mask, cmap="Greys", alpha=alpha)  # Overlay the mask
    #axes[2].set_title("Modern Ortho")

    # Add a common title and adjust layout
    fig.suptitle("Elevation Differences Before and After Alignment", fontsize=16)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 1)
    modern_ortho = np.moveaxis(modern_ortho, 0, 2)
    im3 = axes.imshow(modern_ortho)
    axes.imshow(inlier_mask, cmap="Greys", alpha=alpha)  # Overlay the mask
    axes.set_title("Inlier Mask")
    plt.show()
    """

    # create new transform
    aligned_transform = Affine(
        hist_transform.a,
        hist_transform.b,
        hist_transform.c + aligned_matrix[0, 3],
        hist_transform.d,
        hist_transform.e,
        hist_transform.f + aligned_matrix[1, 3],
    )

    print(hist_transform)
    print(aligned_transform)

    if print_values:
        avg_before = np.ma.mean(diff_before.data)
        med_before = np.ma.median(diff_before.data)
        nmad_before = xdem.spatialstats.nmad(diff_before)

        avg_after = np.ma.mean(diff_after.data)
        med_after = np.ma.median(diff_after.data)
        nmad_after = xdem.spatialstats.nmad(diff_after)

        print(f"Error before (all) - "
              f"mean: {avg_before:.2f}, "
              f"median: {med_before:.2f},"
              f"NMAD = {nmad_before:.2f}")
        print(f"Error after (all) - "
              f"mean: {avg_after:.2f}, "
              f"median:{med_after:.2f}, "
              f"NMAD = {nmad_after:.2f}")

        if inlier_mask is not None:
            inliers_before = diff_before[inlier_mask]
            inliers_avg_before = float(np.ma.mean(inliers_before))
            inliers_med_before = float(np.ma.median(inliers_before))
            inliers_nmad_before = xdem.spatialstats.nmad(inliers_before)

            inliers_after = diff_after[inlier_mask]
            inliers_avg_after = float(np.ma.mean(inliers_after))
            inliers_med_after = float(np.ma.median(inliers_after))
            inliers_nmad_after = xdem.spatialstats.nmad(inliers_after)

            print("--")
            print(f"Error before (inliers) - "
                  f"mean: {inliers_avg_before:.2f}, "
                  f"median = {inliers_med_before:.2f}, "
                  f"NMAD = {inliers_nmad_before:.2f}")
            print(f"Error after (inliers) - "
                  f"mean: {inliers_avg_after:.2f}, "
                  f"median = {inliers_med_after:.2f}, "
                  f"NMAD = {inliers_nmad_after:.2f}")

    output_arr = aligned_dem.data
    output_arr[input_historic_dem == no_data_val] = no_data_val
    output_arr[np.isnan(output_arr)] = no_data_val
    #di.display_images([historic_dem.data, modern_dem.data, output_arr])

    return output_arr, aligned_transform, max_slope

if __name__ == "__main__":
    glaciers = [
        "aagaard_glacier",
        "archer_glacier",
        "attlee_glacier",
        "bevin_glacier",
        "bolton_glacier",
        "centurian_glacier",
        "clarke_glacier",
        "crane_glacier",
        "eden_glacier",
        "evans_glacier",
        "flask_glacier",
        "forbes_glacier",
        "getman_ice_piedmont",
        "hektoria_glacier",
        "hooper_glacier",
        "hurley_glacier",
        "iliad_glacier",
        "jones_is",
        "jorum_glacier",
        "leonardo_glacier",
        "leppard_glacier",
        "mamut_glacier",
        "mapple_glacier",
        "mcmorrin_glacier",
        "melville_glacier",
        "miethe_glacier",
        "mitterling_glacier",
        "montiel_glacier",
        "morrison_glacier",
        "nemo_glacier",
        "neny_glacier",
        "niepce_glacier",
        "northeast_glacier",
        "orel_ice_fringe",
        "pequod_glacier",
        "perutz_glacier",
        "petzval_glacier",
        "rachel_glacier",
        "reid_glacier",
        "seller_glacier",
        "sikorsky_glacier",
        "snowshoe_glacier",
        "somigliana_glacier",
        "starbuck_glacier",
        "stubb_glacier",
        "swithinbank_glacier",
        "thunder_glacier",
        "vogel_glacier",
        "william_glacier"
    ]
    for project_name in glaciers:

        import os
        project_folder = f"/data/ATM/data_1/sfm/agi_projects/{project_name}"
        output_fld = os.path.join(project_folder, "output")
        path_hist_dem = os.path.join(output_fld, project_name + "_dem_absolute.tif")

        import src.load.load_image as li
        dem_abs, transform_abs = li.load_image(path_hist_dem,
                                               return_transform=True)
        old_shape = dem_abs.shape

        import src.base.calc_bounds as cb
        bounds = cb.calc_bounds(transform_abs, dem_abs.shape)

        import src.load.load_rema as lr
        print(bounds)
        dem_modern, transform_rema = lr.load_rema(bounds,
                                                  zoom_level=10,
                                                  auto_download=True,
                                                  return_transform=True)

        # resize the old dem to the modern dem shape
        import src.base.resize_image as rei
        dem_abs_resized = rei.resize_image(dem_abs, dem_modern.shape)

        # update the transform as well
        from affine import Affine

        transform_abs_resized = Affine(
            transform_rema.a,  # New x-scale
            transform_abs.b,  # Original x-skew
            transform_abs.c,  # Original x-offset
            transform_abs.d,  # Original y-skew
            transform_rema.e,  # New y-scale
            transform_abs.f  # Original y-offset
        )

        import src.load.load_rock_mask as lrm
        mask_resolution = 10
        rock_mask = lrm.load_rock_mask(bounds, mask_resolution, mask_buffer=10)
        print(dem_abs_resized.shape, dem_modern.shape)
        print(transform_abs, transform_rema)

        try:
            dem_corrected, transform_corrected, new_max_slope = correct_dem(dem_abs_resized, dem_modern,
                                           transform_abs_resized, transform_rema,
                                           mask=rock_mask,
                                           adapt_mask=True,
                                           max_slope=20)
        except:
            print("CORRECTION FAILED FOR ", project_name)
            input()

        # resize back to the original shape
        dem_corrected = rei.resize_image(dem_corrected, old_shape)

        # adapt transform to reflect original pixel size
        transform_corrected = Affine(
            transform_abs.a,
            transform_corrected.b,
            transform_corrected.c,
            transform_corrected.d,
            transform_abs.e,
            transform_corrected.f
        )

        # adapt output path
        output_fld = "/data/ATM/data_1/papers/paper_sfm/finished_dems_corrected2/"
        output_path_dem_corr = output_fld + project_name + "_dem_corrected.tif"

        import src.export.export_tiff as eti
        eti.export_tiff(dem_corrected, output_path_dem_corr,
                        transform=transform_corrected, overwrite=True,
                        use_lzw=True, no_data=-9999)
