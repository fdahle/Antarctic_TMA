import copy
import os

import numpy as np

import src.load.load_image as li
import src.load.load_rema as lr
import src.load.load_rock_mask as lrm
import src.base.calc_bounds as cb
import src.base.resize_image as ri
import src.display.display_images as di

project_name = "new_tst8"
path_project_fld = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/output/"

# load historic dem
path_hist_dem = os.path.join(path_project_fld, f"{project_name}_dem_absolute.tif")
hist_dem, transform = li.load_image(path_hist_dem, return_transform=True)

# get bounds
bounds = cb.calc_bounds(transform, hist_dem.shape)

# load modern dem
modern_dem, _ = lr.load_rema(bounds)

orig = copy.deepcopy(hist_dem)

# resize both images to the smaller one
hist_dem = ri.resize_image(hist_dem, modern_dem.shape)

modern_dem[hist_dem == -9999] = np.amin(modern_dem) - 1
hist_dem[hist_dem == -9999] = np.amin(hist_dem) - 1

from skimage.feature import peak_local_max
hist_peak_coords = peak_local_max(hist_dem, min_distance=50)
modern_peak_coords = peak_local_max(modern_dem, min_distance=50)

hist_peak_coords = np.flip(hist_peak_coords, axis=1)
modern_peak_coords = np.flip(modern_peak_coords, axis=1)

# load the rockmask
rock_mask = lrm.load_rock_mask(bounds, resolution=10)
rock_mask = ri.resize_image(rock_mask, hist_dem.shape)

# filter out points where the rock mask is 0
mask = rock_mask == 1
hist_peak_coords = hist_peak_coords[mask[hist_peak_coords[:, 1], hist_peak_coords[:, 0]]]
modern_peak_coords = modern_peak_coords[mask[modern_peak_coords[:, 1], modern_peak_coords[:, 0]]]

from scipy.spatial import cKDTree

# Set a distance threshold to disregard unmatched points
distance_threshold = 500  # adjust based on your data

# Build a KDTree for efficient nearest-neighbor search
modern_tree = cKDTree(modern_peak_coords)

# Find the closest point in modern_dem_coords for each point in hist_peak_coords
distances, indices = modern_tree.query(hist_peak_coords, distance_upper_bound=distance_threshold)

# Sort matches by distance
sorted_indices = np.argsort(distances)
distances = distances[sorted_indices]
hist_peak_coords = hist_peak_coords[sorted_indices]
indices = indices[sorted_indices]

# Track unique matches
matched_pairs = []
used_modern_indices = set()

for i, (distance, hist_point, modern_idx) in enumerate(zip(distances, hist_peak_coords, indices)):
    # Ignore unmatched points
    if distance == np.inf or modern_idx in used_modern_indices:
        continue

    # Add the match and mark the modern index as used
    matched_pairs.append([hist_point[0], hist_point[1],
                          modern_peak_coords[modern_idx][0], modern_peak_coords[modern_idx][1]])
    used_modern_indices.add(modern_idx)

# Convert matched pairs to a numpy array
matched_coords = np.array(matched_pairs)

tmp = copy.deepcopy(hist_dem)

hist_dem[tmp == -9999] = np.nan
modern_dem[tmp == -9999] = np.nan

hist_dem[rock_mask == 0] = np.nan
modern_dem[rock_mask == 0] = np.nan

# get the last two columns
tmp_coords = matched_coords[:, 2:]

# convert px coordinates of peaks to absolute coordinates
ones = np.ones((tmp_coords.shape[0], 1))
hist_px_peaks = np.hstack([tmp_coords, ones])
transform = np.asarray(transform)
transform = transform.reshape(3, 3)
a_peaks = hist_px_peaks @ transform.T
absolute_coords = a_peaks[:, :2]

absolute_matches = np.hstack([absolute_coords, matched_coords[:, :2]])

# calc new transform
import src.georef.snippets.calc_transform as ct
import src.georef.snippets.apply_transform as at

new_transform, res = ct.calc_transform(hist_dem, absolute_matches)
print(transform)
print(new_transform)
print(orig.shape)
at.apply_transform(orig, new_transform,
                   save_path = "/home/fdahle/Desktop/tmp/transformed_dem.tif")

path_hist_ortho = os.path.join(path_project_fld, f"{project_name}_ortho_absolute.tif")
ortho = li.load_image(path_hist_ortho)
ortho = ortho[0, :, :]

print(ortho.shape)

at.apply_transform(ortho, new_transform,
                   save_path = "/home/fdahle/Desktop/tmp/transformed_ortho.tif")

