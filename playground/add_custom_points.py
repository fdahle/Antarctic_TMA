import Metashape
import os

project_fld = "/data/ATM/data_1/sfm/agi_projects/"
#project_name = "another_matching_test_gpt"
project_name = "testtest2"

# create the path to the project
project_path = os.path.join(project_fld, project_name, project_name + ".psx")

img_folder = os.path.join(project_fld, project_name, 'data', 'images')
mask_folder = os.path.join(project_fld, project_name, 'data', 'masks_adapted')

doc = Metashape.Document(read_only=False)  # noqa
doc.open(project_path, ignore_lock=True)
chunk = doc.chunk

import playground.find_tps_agi as fta

#tie_points_data = fta.find_tps_agi(img_folder, mask_folder, "sequential")

# Assuming you have a list of tie points with their projections in images
tie_points_data = [
    {
        'projections': {
            'image1.jpg': (1, 1),
            'image2.jpg': (2, 2),
            # ... more images where this tie point is visible
        }
    },
    # ... more tie points
]

#chunk.tie_points = Metashape.TiePoints()
print(chunk.tie_points.points)
#for point in chunk.tie_points.points:
#    print(point)

tps = Metashape.TiePoints()
for tp in tie_points_data:
    # create a tie point
    point = Metashape.TiePoints.Point()
