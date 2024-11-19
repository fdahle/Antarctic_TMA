project_name = "grf_test2"
#project_name = "new_tst8"
import os
import Metashape
PROJECT_FOLDER = "/data/ATM/data_1/sfm/agi_projects"
project_fld = os.path.join(PROJECT_FOLDER, project_name)
project_psx_path = os.path.join(project_fld, project_name + ".psx")

# path to cleaned absolute pc
cleaned_pc_path = os.path.join(project_fld, "output",
                               f"{project_name}_pointcloud_absolute_cleaned.ply")

import src.load.load_ply as lp
points = lp.load_ply(cleaned_pc_path)

# get first 10 points
points = points[:10]

# print coords
print(points[:, 0:3])
