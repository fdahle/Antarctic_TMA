import pandas as pd
import Metashape

project_name = "test_gcps2"

# load the gcps
path_gcps = f"/data/ATM/data_1/sfm/agi_projects/{project_name}/{project_name}_gcps.csv"
gcps = pd.read_csv(path_gcps, sep=';')


doc = Metashape.Document()
doc.open(f"/data/ATM/data_1/sfm/agi_projects/{project_name}/{project_name}.psx")

chunk = doc.chunks[0]
ortho = chunk.orthomosaic
dem = chunk.elevation

print(ortho.left, ortho.top, ortho.resolution)
print(dem.left, dem.top, dem.resolution)
print(dem.width, dem.height)
print(ortho.width, ortho.height)

import pandas as pd

new_lst = []

from tqdm import tqdm

ttl = gcps.shape[0] * len(chunk.cameras)
pbar = tqdm(total=ttl)

# iterate over the relative coordinates
for _, row in gcps.iterrows():

	# remove the marker if it is already existing
	mrk = [marker for marker in chunk.markers if marker.label == row['GCP']]

	chunk.markers.remove(mrk[0])

	# Convert to geographic coordinates
	#geo_x = ortho.left + rel_coords[0] * ortho.resolution
	#geo_y = ortho.top - rel_coords[1] * ortho.resolution

	x = row['x_rel'] / ortho.width * dem.width
	y = row['y_rel'] / ortho.height * dem.height

	print(ortho.resolution / dem.resolution)

	x = x * dem.resolution / ortho.resolution
	y = y * dem.resolution / ortho.resolution

	print("X", row['x_rel'],x)


	# Convert pixel coordinates to geographic coordinates (ortho)
	geo_x = dem.left + x * dem.resolution
	geo_y = dem.top - y * dem.resolution

	#print(rel_coords, rel_coords[0]* dem.resolution, rel_coords[1] * dem.resolution, "GEO", geo_x, geo_y)

	z = dem.altitude([geo_x, geo_y])

	# up until here it's righ

	point_3d = Metashape.Vector([geo_x, geo_y, z])

	point_local = chunk.transform.matrix.inv().mulp(point_3d)

	marker = None
	for camera in chunk.cameras:
		pbar.update(1)
		projection = camera.project(point_local)
		if projection is None:
			continue
		x,y = projection
		if (0 <= x <= camera.image().width) and (0 <= y <= camera.image().height):

			if marker is None:
				marker = chunk.addMarker()
				marker.label = row['GCP']
				marker.reference.location = Metashape.Vector([row['x_abs'], row['y_abs'], 0])
			m_proj = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)
			marker.projections[camera] = m_proj

pbar.close()

doc.save()

	#dem_gcps.append([geo_x, geo_y, z])
	#print(point_3d)


print(new_lst)
exit()

# save dem_gcps as shape points
#import geopandas as gpd
#from shapely.geometry import Point

#points = [Point(x, y) for x, y, z in dem_gcps]
#df = gpd.GeoDataFrame({
#	'geometry': points,
#	'z': [z for x, y, z in dem_gcps]
#})
#output_path = "/data/ATM/data_1/sfm/agi_projects/test_gcps/tst.shp"
#df.to_file(output_path)


