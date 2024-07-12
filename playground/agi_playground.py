import Metashape
import src.base.load_credentials as lc

# get the license key
licence_key = lc.load_credentials("agisoft")['licence']

# Activate the license
Metashape.License().activate(licence_key)

# create Metashape document and open the psx
path_psx = "/data/ATM/data_1/sfm/agi_projects/test_gcps/test_gcps.psx"
doc = Metashape.Document()
doc.open(path_psx)

# load the chunk
chunk = doc.chunks[0]

"""
center = chunk.region.center
size = chunk.region.size

# Calculate the minimum and maximum corners of the bounding box
min_corner = Metashape.Vector([center.x - size.x / 2, center.y - size.y / 2])
max_corner = Metashape.Vector([center.x + size.x / 2, center.y + size.y / 2])

# Create the bounding box
bounding_box = Metashape.BBox(min_corner, max_corner)

projection = Metashape.OrthoProjection()
projection.crs = chunk.crs

print(type(bounding_box))
print(bounding_box.max, bounding_box.min)

chunk.buildDem(source_data=Metashape.DataSource.PointCloudData,
               region=bounding_box,
               projection=projection,
               interpolation=Metashape.Interpolation.EnabledInterpolation)

output_dem_path = "/home/fdahle/Desktop/agi_test/output3/dem_relative.tif"
export_params = {
    'path': output_dem_path,
    'source_data': Metashape.ElevationData,
    'image_format': Metashape.ImageFormatTIFF,
    'raster_transform': Metashape.RasterTransformNone,
    'region': bounding_box,
    'nodata_value': -9999,
    'resolution': 0.001
}
chunk.exportRaster(**export_params)
"""

"""
for point in csv_points:
    point2D = PhotoScan.Vector([imgX, imgY])  # coordinates of the point on the given photo

chunk.addMarker()
"""