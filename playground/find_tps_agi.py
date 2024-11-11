import src.base.find_tie_points as ftp
import src.display.display_images as di
import src.load.load_image as li
import src.base.rotate_image as ri

project_name = "big_tst"
project_folder = f"/data/ATM/data_1/sfm/agi_projects/{project_name}"
image_folder = project_folder + "/data/images_enhanced"
mask_folder = project_folder + "/data/masks_adapted"

min_conf = 0.9
tp_type = float
debug_show_intermediate_steps = False

image_id_1 = "CA213732V0072"
image_id_2 = "CA213733R0072"

# load the images
image_1 = li.load_image(image_folder + f"/{image_id_1}.tif")
image_2 = li.load_image(image_folder + f"/{image_id_2}.tif")

# load the masks
mask_1 = li.load_image(mask_folder + f"/{image_id_1}_mask.tif")
mask_2 = li.load_image(mask_folder + f"/{image_id_2}_mask.tif")

image_1 = ri.rotate_image(image_1, 180)
mask_1 = ri.rotate_image(mask_1, 180)
image_2 = ri.rotate_image(image_2, 180)
mask_2 = ri.rotate_image(mask_2, 180)

# init tie point detector
tpd = ftp.TiePointDetector('lightglue', verbose=True,
                           min_conf_value=min_conf, tp_type=tp_type,
                           display=debug_show_intermediate_steps)

tps, conf = tpd.find_tie_points(image_1, image_2, mask_1, mask_2)

# display the images
di.display_images([image_1, image_2], tie_points=tps, tie_points_conf=conf)