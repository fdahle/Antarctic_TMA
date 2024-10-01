import os.path
import Metashape
import math
import statistics

project_fld = "/data/ATM/data_1/sfm/agi_projects/"
project_name = "test_focal_relative"

# create the path to the project
project_path = os.path.join(project_fld, project_name, project_name + ".psx")

doc = Metashape.Document(read_only=False)  # noqa
doc.open(project_path, ignore_lock=True)
chunk = doc.chunk

point_cloud = chunk.tie_points
points = point_cloud.points
error, tie_points = [], []

for camera in [cam for cam in doc.chunk.cameras if cam.transform]:
    print(camera.label)
    point_index = 0
    photo_num = 0
    error_image = []
    for proj in doc.chunk.tie_points.projections[camera]:
        track_id = proj.track_id
        while point_index < len(points) and points[point_index].track_id < track_id:
            point_index += 1
        if point_index < len(points) and points[point_index].track_id == track_id:
            if not points[point_index].valid:
                continue

            dist = camera.error(points[point_index].coord, proj.coord).norm() ** 2
            error.append(dist)
            error_image.append(dist)

            photo_num += 1
    err = round(math.sqrt(sum(error_image) / len(error_image)), 2)
    print(err)

    tie_points.append(photo_num)

reprojection_rmse = round(math.sqrt(sum(error) / len(error)), 2)
reprojection_max = round(max(error), 2)
reprojection_std = round(statistics.stdev(error), 2)
tie_points_per_image = round(sum(tie_points) / len(tie_points), 0)

print("Average tie point residual error: " + str(reprojection_rmse))
print("Maxtie point residual error: " + str(reprojection_max))
print("Standard deviation for tie point residual error: " + str(reprojection_std))
print("Average number of tie points per image: " + str(tie_points_per_image))

"""
import os.path
import Metashape

project_fld = "/data/ATM/data_1/sfm/agi_projects/"
project_name = "test_focal_relative"

# create the path to the project
project_path = os.path.join(project_fld, project_name, project_name + ".psx")

doc = Metashape.Document(read_only=False)  # noqa
doc.open(project_path, ignore_lock=True)
chunk = doc.chunk

point_cloud = chunk.tie_points
projections = point_cloud.projections
points = point_cloud.points
npoints = len(points)
tracks = point_cloud.tracks
point_ids = [-1] * len(point_cloud.tracks)

for point_id in range(0, npoints):
	point_ids[points[point_id].track_id] = point_id


for camera in chunk.cameras:
	nprojections = 0

	if camera.type == Metashape.Camera.Type.Keyframe:

		continue # skipping Keyframes

	if not camera.transform:
		continue

	for proj in projections[camera]:
		track_id = proj.track_id
		point_id = point_ids[track_id]
		if point_id < 0:
			continue
		if not points[point_id].valid:
			continue

		nprojections += 1

	print(camera, nprojections, len(projections[camera]))
"""