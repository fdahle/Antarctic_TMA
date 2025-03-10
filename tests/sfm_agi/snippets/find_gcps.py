project_name = "wellman_glacier"

# define project path
project_path = "/data/ATM/data_1/sfm/agi_projects/" + project_name

# load ortho_rel
import src.load.load_image as li
ortho_rel = li.load_image(project_path + f"/output/{project_name}_ortho_relative.tif")
# remove alpha channel
if len(ortho_rel.shape) == 3:
    ortho_rel = ortho_rel[0, :, :]

# load best rotation
path_best_rpt = project_path + "/data/georef/best_rot.txt"
import numpy as np
best_rot = np.loadtxt(path_best_rpt)
best_rot = float(best_rot)

print(best_rot)

# load transform
import src.load.load_transform as lt
transform_path = project_path + "/data/georef/transform.txt"
transform_georef = lt.load_transform(transform_path, delimiter=",")
import src.base.calc_bounds as cb
bounds_georef_old = cb.calc_bounds(transform_georef, ortho_rel.shape)
print(bounds_georef_old)

# load satellite image
import src.load.load_satellite as ls
ortho_new = ls.load_satellite(bounds_georef_old)



# rotate image
import src.base.rotate_image as ri
ortho_rel = ri.rotate_image(ortho_rel, best_rot)

import src.base.resize_image as rei
ortho_rel = rei.resize_image(ortho_rel, (7678, 9044))

# resize to sat size
ortho_rel = rei.resize_image(ortho_rel, ortho_new.shape[-2:])


import src.display.display_images as di
di.display_images([ortho_rel, ortho_new, ortho_new],
                  overlays=[None, None, ortho_rel])
