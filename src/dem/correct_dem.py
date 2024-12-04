# Library imports
import numpy as np
import xdem
from affine import Affine

# Local imports
import src.base.resize_image as ri


def correct_dem(historic_dem, modern_dem, hist_transform, modern_transform,
                historic_ortho=None, modern_ortho=None,
                mask=None, max_slope=None, no_data_val=-9999, print_values=True):

    # check if the DEMs have the same shape
    if historic_dem.shape != modern_dem.shape:
        raise ValueError("The two DEMs have different shapes")

    # update mask with no data values
    if mask is not None:
        mask[historic_dem == no_data_val] = 0

    # set no data values to nan
    historic_dem[historic_dem == no_data_val] = np.nan

    #import matplotlib.pyplot as plt
    #fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #axes[0].imshow(historic_dem)
    #axes[1].imshow(mask)
    #plt.title("mask")
    #plt.tight_layout()
    #plt.show()

    # get mask as numpy array and resize to modern dem shape
    if mask is not None:
        mask = ri.resize_image(mask, modern_dem.shape)

    # modern_ortho = ri.resize_image(modern_ortho, modern_dem.shape)

    print(modern_dem.shape, modern_ortho.shape, mask.shape)

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
                slope_mask = slope < max_slope  # Convert max_slope to radians
                slope_mask = slope_mask.data
                slope_mask = np.ma.getdata(slope_mask)

            # create a mask for the stable ground
            if mask is not None:
                inlier_mask = mask & slope_mask
            else:
                inlier_mask = slope_mask

            # invert mask
            inlier_mask = inlier_mask.astype(bool)

            #import matplotlib.pyplot as plt
            #plt.imshow(inlier_mask)
            #plt.title("Inlier mask")
            #plt.show()

            print("Fit deramp")
            deramp.fit(modern_dem, historic_dem, inlier_mask, verbose=True)

            #print("Apply deramp")
            aligned_dem = deramp.apply(historic_dem)

            # fit the co-registration
            print("Fit Nuth Kääb")
            nuth_kaab.fit(modern_dem, aligned_dem, inlier_mask, verbose=True)

            # apply the co-registration
            print("Apply Nuth Kääb")
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

    import matplotlib.pyplot as plt
    vmin, vmax= -50, 50
    alpha = 0.2
    # Create a figure and two subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    # Create a figure and two subplots
    im1 = axes[0].imshow(diff_before.data, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
    axes[0].imshow(inlier_mask, cmap="Greys", alpha=alpha)  # Overlay the mask
    axes[0].set_title("Difference Before")
    plt.colorbar(im1, ax=axes[0], label="Elevation change (m)")

    # Plot diff_after on the second subplot
    im2 = axes[1].imshow(diff_after.data, cmap="coolwarm_r", vmin=vmin, vmax=vmax)
    axes[1].imshow(inlier_mask, cmap="Greys", alpha=alpha)  # Overlay the mask
    axes[1].set_title("Difference After")
    plt.colorbar(im2, ax=axes[1], label="Elevation change (m)")

    # put first axis of modern ortho in the last
    modern_ortho = np.moveaxis(modern_ortho, 0, 2)
    im3 = axes[2].imshow(modern_ortho)
    axes[2].imshow(inlier_mask, cmap="Greys", alpha=alpha)  # Overlay the mask
    axes[2].set_title("Modern Ortho")

    # Add a common title and adjust layout
    fig.suptitle("Elevation Differences Before and After Alignment", fontsize=16)
    plt.tight_layout()
    plt.show()

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
    output_arr[historic_dem.data == no_data_val] = no_data_val

    return output_arr, aligned_transform, max_slope

"""if __name__ == "__main__":

    project_name = "test3"

    import os
    project_folder = f"/data/ATM/data_1/sfm/agi_projects/{project_name}"
    output_fld = os.path.join(project_folder, "output")
    path_hist_dem = os.path.join(output_fld, project_name + "_dem_absolute.tif")

    import src.load.load_image as li
    dem_abs, transform_abs = li.load_image(path_hist_dem,
                                           return_transform=True)

    import src.base.calc_bounds as cb
    bounds = cb.calc_bounds(transform_abs, dem_abs.shape)

    import src.load.load_rema as lr
    print(bounds)
    dem_modern, transform_rema = lr.load_rema(bounds,
                                              output_resolution=2,
                                              auto_download=True,
                                              return_transform=True)

    #import src.load.load_rock_mask as lrm
    #mask_resolution = 10
    #rock_mask = lrm.load_rock_mask(bounds, mask_resolution)

    print(dem_abs.shape, dem_modern.shape)
    print(transform_abs, transform_rema)

    import src
    exit()

    dem_corrected = correct_dem(dem_abs, dem_modern,
                                   transform_abs, transform_rema,
                                   mask=rock_mask,
                                   max_slope=20)
"""
