import os.path

import Metashape
import numpy as np

import src.base.connect_to_database as ctd
import src.base.find_tie_points as ftp
import src.base.rotate_image as ri
import src.display.display_images as di
import src.load.load_image as li

agi_project = "agi_tps"
id_image_1 = "CA184332V0069"
id_image_2 = "CA184332V0071"

debug_show_intermediate_steps = True

use_rotation = True

min_conf = 0.9
tp_type = float

conn = ctd.establish_connection()

project_fld = f"/data/ATM/data_1/sfm/agi_projects/"
project_psx_path = os.path.join(project_fld, agi_project, agi_project + ".psx")
data_fld = os.path.join(project_fld, agi_project, "data")

image_path = os.path.join(data_fld, "images")
image_enhanced_path = os.path.join(data_fld, "images_enhanced")
mask_path = os.path.join(data_fld, "masks")
mask_adapted_path = os.path.join(data_fld, "masks_adapted")
if os.path.isdir(image_enhanced_path):
    image_path = image_enhanced_path
if os.path.isdir(mask_adapted_path):
    mask_path = mask_adapted_path

# get the images and masks
doc = Metashape.Document()
doc.open(project_psx_path, ignore_lock=True)
chunk = doc.chunk

# get idx of image 1
for i, camera in enumerate(chunk.cameras):
    if camera.label == id_image_1:
        idx_image_1 = i
for i, camera in enumerate(chunk.cameras):
    if camera.label == id_image_2:
        idx_image_2 = i

print(f"Load images from {image_path}")

image1 = li.load_image(os.path.join(image_path, id_image_1 + ".tif"))
image2 = li.load_image(os.path.join(image_path, id_image_2 + ".tif"))

mask1 = li.load_image(os.path.join(mask_path, id_image_1 + "_mask.tif"))
mask2 = li.load_image(os.path.join(mask_path, id_image_2 + "_mask.tif"))


if use_rotation:
    # get the rotation of the images
    sql_string = ("SELECT azimuth_exact FROM images_extracted WHERE "
                  "image_id in ('" + id_image_1 + "', '" + id_image_2 + "')")
    data = ctd.execute_sql(sql_string, conn)

    # get the rotation angles
    rot1 = data['azimuth_exact'][0]
    rot2 = data['azimuth_exact'][1]

    # account for the different coordinate system
    rot1 = 360 - rot1 + 90
    rot1 = round(rot1, 2)
    rot2 = 360 - rot2 + 90
    rot2 = round(rot2, 2)
else:
    rot1 = 0
    rot2 = 0

# init tie-point detector
tpd = ftp.TiePointDetector('lightglue', verbose=True,
                           min_conf_value=min_conf, tp_type=tp_type,
                           display=debug_show_intermediate_steps)

if rot1 != rot2:

    # rotate the images
    image1, rotmat1 = ri.rotate_image(image1, rot1, return_rot_matrix=True)
    image2, rotmat2 = ri.rotate_image(image2, rot2, return_rot_matrix=True)

    # rotate the masks
    if mask1 is not None:
        mask1 = ri.rotate_image(mask1, rot1)
    if mask2 is not None:
        mask2 = ri.rotate_image(mask2, rot2)

tps, conf = tpd.find_tie_points(image1, image2,
                                mask1=mask1, mask2=mask2)

# make empty tps at least 2d
if tps.shape[0] == 0:
    tps = np.zeros((0, 4))
    conf = np.zeros((0))

style_config = {
    'title': f"{tps.shape[0]} tie points between {id_image_1} ({rot1}) and {id_image_2} ({rot2})",
    'line_width': 0.2,
}

print(tps.shape, conf.shape)

di.display_images([image1, image2], tie_points=tps, tie_points_conf=conf)
