import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt

import xdem

output_fld = "/data/ATM/data_1/sfm/agi_projects/new_test_3/output/"
path_old_dem = output_fld + "new_test_3_dem_absolute.tif"

import src.load.load_image as li
old_dem, old_transform = li.load_image(path_old_dem, return_transform=True)
old_dem[old_dem == -9999] = np.nan

import src.base.calc_bounds as cb
bounds = cb.calc_bounds(old_transform, old_dem.shape)
import src.load.load_rema as lr
new_dem = lr.load_rema(bounds)

import src.base.resize_image as ri
new_dem = ri.resize_image(new_dem, old_dem.shape)
new_dem[old_dem == np.nan] = np.nan

import src.load.load_rock_mask as lrm
rock_mask = lrm.load_rock_mask(bounds, return_shapes=True)

# set crs of rock_mask to 3031
rock_mask.crs = 3031

reference_dem = xdem.DEM.from_array(new_dem, transform=old_transform, crs=3031)
dem_to_be_aligned = xdem.DEM.from_array(old_dem, transform=old_transform, crs=3031)
rock_outlines = gu.Vector(rock_mask)

if rock_outlines.crs != reference_dem.crs:
    rock_outlines = rock_outlines.to_crs(reference_dem.crs)

# Create a stable ground mask (not glacierized) to mark "inlier data"
inlier_mask = rock_outlines.create_mask(reference_dem)

#import src.display.display_images as di
#di.display_images([new_dem, old_dem], overlays=[inlier_mask.data, inlier_mask.data],)

diff_before = reference_dem - dem_to_be_aligned
diff_data = diff_before.data

# Flatten the data array and exclude NaN values
valid_data = np.copy(diff_data[~np.isnan(diff_data)])  # Ensure the array is writable

# Calculate percentiles
vmin = np.percentile(valid_data, 2)
vmax = np.percentile(valid_data, 98)

# Create a symmetric range around 0
max_abs = max(abs(vmin), abs(vmax))
vmin = -max_abs
vmax = max_abs

# Create a colormap centered at 0
import matplotlib.colors as mcolors
cmap = plt.cm.coolwarm_r
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)


# Create a plot using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(diff_data, cmap=cmap, norm=norm)
plt.colorbar(label="Elevation change (m)")
plt.title("Elevation Difference Before Co-Registration")
plt.show()

nuth_kaab = xdem.coreg.NuthKaab()

nuth_kaab.fit(reference_dem, dem_to_be_aligned, inlier_mask)

aligned_dem = nuth_kaab.apply(dem_to_be_aligned)

import src.export.export_tiff as et
et.export_tiff(aligned_dem.data, output_fld + "aligned_dem.tif", overwrite=True,
               transform=old_transform)

diff_after = reference_dem - aligned_dem
diff_after_data = diff_after.data

# Create a plot using matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(diff_after_data, cmap=cmap, norm=norm)
plt.colorbar(label="Elevation change (m)")
plt.title("Elevation Difference after Co-Registration")
plt.show()

inliers_before = diff_before[inlier_mask]
med_before, nmad_before = np.median(inliers_before), xdem.spatialstats.nmad(inliers_before)

inliers_after = diff_after[inlier_mask]
med_after, nmad_after = np.median(inliers_after), xdem.spatialstats.nmad(inliers_after)

print(f"Error before: median = {med_before:.2f} - NMAD = {nmad_before:.2f} m")
print(f"Error after: median = {med_after:.2f} - NMAD = {nmad_after:.2f} m")
