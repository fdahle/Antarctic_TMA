from src.sfm_agi.run_agi import resolution_abs

project_name = "test3"
#project_name = "new_tst8"
import os
import Metashape
PROJECT_FOLDER = "/data/ATM/data_1/sfm/agi_projects"
project_fld = os.path.join(PROJECT_FOLDER, project_name)
project_psx_path = os.path.join(project_fld, project_name + ".psx")

doc = Metashape.Document()
doc.open(project_psx_path, ignore_lock=True)
chunk = doc.chunk

center = chunk.region.center
size = chunk.region.size
rotate = chunk.region.rot
T = chunk.transform.matrix

corners = [T.mulp(center + rotate * Metashape.Vector(
    [size[0] * ((i & 1) - 0.5), 0.5 * size[1] * ((i & 2) - 1), 0.25 * size[2] * ((i & 4) - 2)])) for i in range(8)]

if chunk.crs:
    corners = [chunk.crs.project(x) for x in corners]

# get min and max values
min_x = min(corners, key=lambda x: x[0])[0]
max_x = max(corners, key=lambda x: x[0])[0]
min_y = min(corners, key=lambda x: x[1])[1]
max_y = max(corners, key=lambda x: x[1])[1]

print("min_x: ", min_x)
print("max_x: ", max_x)
print("min_y: ", min_y)
print("max_y: ", max_y)

import xdem
glacier_outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
mask_glacier = glacier_outlines.create_mask(dh)
