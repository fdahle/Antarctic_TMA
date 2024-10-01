import os
import Metashape

project_fld = "/data/ATM/data_1/sfm/agi_projects/"
project_name = "another_matching_try_agi"

# create the path to the project
project_path = os.path.join(project_fld, project_name, project_name + ".psx")

doc = Metashape.Document(read_only=False)  # noqa
doc.open(project_path, ignore_lock=True)
chunk = doc.chunk

points = chunk.tie_points.Points

print(chunk.tie_points)
print(chunk.tie_points.tracks)
for i, track in enumerate(chunk.tie_points.tracks):
    print(i, track.color)
