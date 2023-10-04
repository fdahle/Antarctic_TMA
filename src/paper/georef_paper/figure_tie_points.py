import numpy as np

import base.load_image_from_file as liff
import base.remove_borders as rb

import image_tie_points.find_tie_points as ftp

import display.display_images as di
import display.display_tiepoints as dt

#id1 = "CA182332V0095"
id1 = "CA216632V0282"
id2 = "CA216632V0283"

# get information from table images
#sql_string = f"SELECT * FROM images WHERE image_id='{id1}'"
#data_images = ctd.get_data_from_db(sql_string, catch=False, verbose=False)

# get information from table images_extracted
#sql_string = "SELECT image_id, height, focal_length, complexity, " \
#             "ST_AsText(footprint_approx) AS footprint_approx, " \
#             "ST_AsText(footprint) AS footprint, " \
#             "footprint_type AS footprint_type " \
#             f"FROM images_extracted WHERE image_id ='{id1}'"
#data_extracted = ctd.get_data_from_db(sql_string, catch=False, verbose=False)

# merge the data from both tables
#data = pd.merge(data_images, data_extracted, on='image_id').iloc[0]

#poly = shapely.from_wkt(data["footprint"])
#min_x, min_y, max_x, max_y = poly.bounds
#sat_bounds = [min_x, min_y, max_x, max_y]
#sat = lsd.load_satellite_data(sat_bounds)

mask_fld = "/data_1/ATM/data_1/aerial/TMA/masked"

img1 = liff.load_image_from_file(id1)
img2 = liff.load_image_from_file(id2)

#img1 = air.adjust_image_resolution(sat, sat_bounds, img1, poly.bounds)

#img1 = ri.rotate_image(img1, data['azimuth'], start_angle=0)
"""
mask1 = liff.load_image_from_file(id1, mask_fld)
if mask1 is None:

    # get the borders
    _, edge_dims = rb.remove_borders(img1, id1, return_edge_dims=True)

    # create mask and remove borders
    mask1 = np.ones_like(img1)
    mask1[:edge_dims[2], :] = 0
    mask1[edge_dims[3]:, :] = 0
    mask1[:, :edge_dims[0]] = 0
    mask1[:, edge_dims[1]:] = 0

mask2 = liff.load_image_from_file(id2, mask_fld)
if mask2 is None:

    # get the borders
    _, edge_dims = rb.remove_borders(img2, id2, return_edge_dims=True)

    # create mask and remove borders
    mask2 = np.ones_like(img2)
    mask2[:edge_dims[2], :] = 0
    mask2[edge_dims[3]:, :] = 0
    mask2[:, :edge_dims[0]] = 0
    mask2[:, edge_dims[1]:] = 0

import time
start = time.time()

import base.resize_image as ri
img1 = ri.resize_image(img1, (10000, 10000))
mask1 = ri.resize_image(mask1, (10000, 10000))
img2 = ri.resize_image(img2, (10000, 10000))
mask2 = ri.resize_image(mask2, (10000, 10000))

tps, conf = ftp.find_tie_points(img1, img2,
                                mask_1=mask1, mask_2=mask2,
                                additional_matching=True, extra_matching=True,
                                matching_method="SuperGlue",
                                verbose=True, catch=False)
exit()

print(tps.shape)
print(np.mean(conf))

end = time.time()
print(end - start)

dt.display_tiepoints([img1, img2], tps, conf, verbose=True)
exit()

points = tps[:,:2].tolist()
print(points)

#di.display_images([img1], points=[points])

subset_min_x = 6000
subset_max_x = 8000
subset_min_y = 2000
subset_max_y = 4000

tps_subset = tps[
    (tps[:, 0] >= subset_min_x) &
    (tps[:, 0] <= subset_max_x) &
    (tps[:, 1] >= subset_min_y) &
    (tps[:, 1] <= subset_max_y)
]

tps_subset_1[:,0] = tps_subset_1[:,0] - subset_min_x_1
tps_subset_1[:,1] = tps_subset_1[:,1] - subset_min_y_1

subset_points_1 = tps_subset_1[:,:2].tolist()

subset_img_1 = img1[subset_min_y_1:subset_max_y_1, subset_min_x_1:subset_max_x_1]

print(subset_img_1.shape)

di.display_images([subset_img_1], points=[subset_points_1], point_size=10)

subset_min_x_other = np.amin(tps_subset[:,2])
subset_max_x_other = np.amax(tps_subset[:,2])
subset_min_y_other = np.amin(tps_subset[:,3])
subset_max_y_other = np.amax(tps_subset[:,2])

tps_subset[:,2] = tps_subset[:, 2] - subset_min_x_other
tps_subset[:,3] = tps_subset[:, 3] - subset_min_y_other

subset_points_2 = tps_subset[:,2:].tolist()

print(subset_points_2)

print(subset_min_x_other, subset_max_x_other)
print(subset_min_y_other, subset_max_y_other)

subset_img_2 = img2[subset_min_y_other:subset_max_y_other, subset_min_x_other:subset_max_x_other]

#di.display_images([subset_img_2], points=[subset_points_2], point_size=10)

s_width = subset_max_x_other - subset_min_x_other
s_height = subset_max_y_other - subset_min_y_other

#di.display_images([img2], bboxes=[subset_min_x_other, subset_min_y_other, s_width, s_height])

# vertical and horizonatal lines on image:
step_x = 2000
step_y = 2000


vertical_lines = [(x, 0, x, img1.shape[0]) for x in range(0, img1.shape[1], step_x)]

# Generate horizontal lines
horizontal_lines = [(0, y, img1.shape[1], y) for y in range(0, img1.shape[0], step_y)]

# Combine both lists and return the result
lines = vertical_lines + horizontal_lines

di.display_images([img1], points=[points], point_size=25, lines=lines, line_width=8)
"""
id2 = "CA216632V0283"
img2 = liff.load_image_from_file(id2)

step_x, step_y = 2000, 2000
vertical_lines = [(x, 0, x, img1.shape[0]) for x in range(0, img1.shape[1], step_x)]
# Generate horizontal lines
horizontal_lines = [(0, y, img1.shape[1], y) for y in range(0, img1.shape[0], step_y)]

lines = vertical_lines + horizontal_lines

def apply_transformation(point, transformation_matrix):
    x, y = point
    homogeneous_point = np.array([x, y, 1])
    transformed_point = np.dot(transformation_matrix, homogeneous_point).astype(int)
    return tuple(transformed_point[:2])

def transform_line(line, transformation_matrix):
    transformed_line = []
    for x1, y1, x2, y2 in line:
        new_x1, new_y1 = apply_transformation((x1, y1), transformation_matrix)
        new_x2, new_y2 = apply_transformation((x2, y2), transformation_matrix)
        transformed_line.append((new_x1, new_y1, new_x2, new_y2))#
    return transformed_line

trans_mat = np.asarray([[1.05771460e+00, 1.45373651e-01, -4.95531929e+03],
                        [7.30004815e-03, 1.01576513e+00, -3.43772276e+02]])

print(lines)
transformed_lines = transform_line(lines, trans_mat)
print(transformed_lines)

di.display_images([img2], lines=transformed_lines, line_width=6)