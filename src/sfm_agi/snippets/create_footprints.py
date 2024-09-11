import geopandas as gpd
import Metashape
from shapely.geometry import Point, Polygon


def create_footprints(chunk, save_fld):
    points_path = save_fld + "/cameras_aligned.shp"
    footprints_path = save_fld + "/footprints_aligned.shp"
    flat_footprints_path = save_fld + "/footprints_flat.shp"

    camera_positions = []
    camera_footprints = []
    flat_camera_footprints = []

    if chunk.elevation:
        surface = chunk.elevation
    elif chunk.model:
        surface = chunk.model
    elif chunk.point_cloud:
        surface = chunk.point_cloud
    else:
        surface = chunk.tie_points

    chunk_crs = chunk.crs.geoccs
    if chunk_crs is None:
        chunk_crs = Metashape.CoordinateSystem('LOCAL')  # noqa

    # Get the transformation matrix
    transform = chunk.transform.matrix

    for camera in chunk.cameras:

        c_width = camera.photo.image().width
        c_height = camera.photo.image().height

        if camera.transform:  # Check if the camera has a valid transform matrix
            camera_center = camera.center
            world_position = transform.mulp(camera_center)
            world_position = chunk.crs.project(world_position)

            camera_positions.append({
                "camera_label": camera.label,
                "geometry": Point(world_position.x, world_position.y),
                "height": world_position.z
            })

            import math

            # Get rotation matrix and convert to yaw, pitch, roll
            rot_mat = camera.transform.rotation()
            pitch = -math.asin(rot_mat[2, 0])
            roll = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
            yaw = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
            yaw, pitch, roll = math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

            print(yaw, pitch, roll)

            corners = list()
            flat_corners = list()

            for (x, y) in [[0, 0], [c_width - 1, 0], [c_width - 1, c_height - 1],
                           [0, c_height - 1]]:
                ray_origin = camera.unproject(Metashape.Vector([x, y, 0]))  # noqa
                ray_target = camera.unproject(Metashape.Vector([x, y, 1]))  # noqa
                if type(surface) == Metashape.Elevation:
                    dem_origin = transform.mulp(ray_origin)
                    dem_target = transform.mulp(ray_target)

                    dem_origin = Metashape.OrthoProjection.transform(dem_origin, chunk_crs, surface.projection)
                    dem_target = Metashape.OrthoProjection.transform(dem_target, chunk_crs, surface.projection)
                    corner = surface.pickPoint(dem_origin, dem_target)
                    if corner:
                        corner = Metashape.OrthoProjection.transform(corner, surface.projection, chunk_crs)
                        corner = transform.inv().mulp(corner)
                else:
                    corner = surface.pickPoint(ray_origin, ray_target)
                if not corner:
                    corner = chunk.tie_points.pickPoint(ray_origin, ray_target)
                if not corner:
                    break
                corner = chunk.crs.project(transform.mulp(corner))
                corners.append(corner)

                # Calculate intersection with Z = 0 plane
                t = -ray_origin.z / (ray_target.z - ray_origin.z)
                flat_intersection = ray_origin + t * (ray_target - ray_origin)
                flat_intersection = chunk.crs.project(transform.mulp(flat_intersection))

                flat_corners.append(flat_intersection)

            if corners:
                polygon = Polygon([(corner.x, corner.y) for corner in corners])
                camera_footprints.append({
                    "camera_label": camera.label,
                    "geometry": polygon
                })

            if flat_corners:
                flat_polygon = Polygon([(corner.x, corner.y) for corner in flat_corners])
                flat_camera_footprints.append({
                    "camera_label": camera.label,
                    "geometry": flat_polygon
                })

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(camera_positions)
    gdf_footprints = gpd.GeoDataFrame(camera_footprints)
    gdf_flat_footprints = gpd.GeoDataFrame(flat_camera_footprints)

    # Set the coordinate reference system (CRS)
    gdf.set_crs(epsg=3031, inplace=True)  # Change EPSG code as necessary
    gdf_footprints.set_crs(epsg=3031, inplace=True)  # Change EPSG code as necessary
    gdf_flat_footprints.set_crs(epsg=3031, inplace=True)  # Change EPSG code as necessary

    # Save to a shapefile
    #gdf.to_file(points_path)
    gdf_footprints.to_file(footprints_path)
    gdf_flat_footprints.to_file(flat_footprints_path)
