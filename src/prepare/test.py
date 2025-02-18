import numpy as np

image_id = "CA182732V0034"


import src.load.load_image as li
img = li.load_image(image_id)

import src.prepare.correct_image_orientation as cio
needs_rotation = cio.correct_image_orientation(img, image_id)
print(needs_rotation)

import src.base.connect_to_database as ctd
conn = ctd.establish_connection()

sql_string = f"SELECT * FROM images_extracted WHERE image_id = '{image_id}'"
data = ctd.execute_sql(sql_string, conn)

text_bbox_str = data['text_bbox'].values[0]
text_bboxes = text_bbox_str.split(";")
text_bboxes = [
    [int(x) for x in bbox.replace("(", "").replace(")", "").split(",")]
    for bbox in text_bboxes
]
print(text_bboxes)

import src.display.display_images as di
di.display_images(img, bounding_boxes=[text_bboxes])

import src.base.rotate_image as ri
img_rotated, rot_mat = ri.rotate_image(img, 180, return_rot_matrix=True)

import src.base.rotate_points as rp

# Apply rotation to top-left and bottom-right corners
rotated_bboxes = []
for bbox in text_bboxes:
    # Extract top-left and bottom-right points
    top_left = [bbox[0], bbox[1]]
    bottom_right = [bbox[2], bbox[3]]

    # create np array from points
    pts = np.asarray([top_left, bottom_right])

    # Rotate points
    pts_rotated = rp.rotate_points(pts, rot_mat)

    print(pts, pts_rotated)

    # Reconstruct the bounding box from rotated points
    top_left = pts_rotated[1]
    bottom_right = pts_rotated[0]

    rotated_bboxes.append([int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])])

di.display_images(img_rotated, bounding_boxes=[rotated_bboxes])