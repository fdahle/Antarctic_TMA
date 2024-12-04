import os.path
import Metashape
import math
import statistics

project_fld = "/data/ATM/data_1/sfm/agi_projects/"
project_name = "pequod"

# create the path to the project
project_path = os.path.join(project_fld, project_name, project_name + ".psx")

doc = Metashape.Document(read_only=False)  # noqa
doc.open(project_path, ignore_lock=True)
chunk = doc.chunks[0]

for camera in chunk.cameras:
    print(camera.mask.image()[0, 1000])
