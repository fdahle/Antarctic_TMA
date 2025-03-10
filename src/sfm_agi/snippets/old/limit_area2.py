# Creates footprint shape layer in the active chunk.
#
# This is python script for Metashape Pro. Scripts repository: https://github.com/agisoft-llc/metashape-scripts

import Metashape
import multiprocessing
import concurrent.futures


def calib_valid(calib, point):
    reproj = calib.project(calib.unproject(point))
    if not reproj:
        return False
    return (reproj - point).norm() < 1.0


def create_footprints(doc):
    """
    Creates four-vertex shape for each aligned camera (footprint) in the active chunk
    and puts all these shapes to a new separate shape layer
    """

    chunk = doc.chunk

    if not chunk.shapes:
        chunk.shapes = Metashape.Shapes()
        chunk.shapes.crs = chunk.crs
    T = chunk.transform.matrix
    footprints = chunk.shapes.addGroup()
    footprints.label = "Footprints"
    footprints.color = (30, 239, 30)

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
        chunk_crs = Metashape.CoordinateSystem('LOCAL')

    tls = {}
    brs = {}
    bls = {}
    trs = {}

    def process_camera(chunk, camera):
        if camera.type != Metashape.Camera.Type.Regular or not camera.transform:
            return  # skipping NA cameras

        sensor = camera.sensor
        w, h = sensor.width, sensor.height

        if sensor.film_camera:
            if "File/ImageWidth" in camera.photo.meta and "File/ImageHeight" in camera.photo.meta:
                w, h = int(camera.photo.meta["File/ImageWidth"]), int(camera.photo.meta["File/ImageHeight"])
            else:
                image = camera.photo.image()
                w, h = image.width, image.height

            tl = Metashape.Vector((0, 0))
            br = Metashape.Vector((w, h))
            bl = Metashape.Vector((0, h))
            tr = Metashape.Vector((w, 0))
        else:
            if sensor.key in tls:
                tl = tls[sensor.key]
                br = brs[sensor.key]
                bl = bls[sensor.key]
                tr = trs[sensor.key]
            else:
                tl = None
                br = None
                bl = None
                tr = None

                size = max(w, h)
                calibration_stable = True

                for t in range(size // 2):

                    if tl is None:
                        pt = Metashape.Vector([t * (w - 1) // size, t * (h - 1) // size])
                        if calib_valid(sensor.calibration, pt):
                            tl = pt
                        else:
                            calibration_stable = False

                    if br is None:
                        pt = Metashape.Vector([(size - t) * (w - 1) // size, (size - t) * (h - 1) // size])
                        if calib_valid(sensor.calibration, pt):
                            br = pt
                        else:
                            calibration_stable = False

                    if bl is None:
                        pt = Metashape.Vector([t * (w - 1) // size, (size - t) * (h - 1) // size])
                        if calib_valid(sensor.calibration, pt):
                            bl = pt
                        else:
                            calibration_stable = False

                    if tr is None:
                        pt = Metashape.Vector([(size - t) * (w - 1) // size, t * (h - 1) // size])
                        if calib_valid(sensor.calibration, pt):
                            tr = pt
                        else:
                            calibration_stable = False

                if not calibration_stable:
                    print(
                        "Sensor \"" + sensor.label + "\" (" + camera.label + ") calibration is unstable at the corners. Cropping footprints.")

                tls[sensor.key] = tl
                brs[sensor.key] = br
                bls[sensor.key] = bl
                trs[sensor.key] = tr

        corners = list()
        for (x, y) in [tl, tr, br, bl]:
            ray_origin = camera.unproject(Metashape.Vector([x, y, 0]))
            ray_target = camera.unproject(Metashape.Vector([x, y, 1]))

            if type(surface) == Metashape.Elevation:
                dem_origin = T.mulp(ray_origin)
                dem_target = T.mulp(ray_target)
                dem_origin = Metashape.OrthoProjection.transform(dem_origin, chunk_crs, surface.projection)
                dem_target = Metashape.OrthoProjection.transform(dem_target, chunk_crs, surface.projection)
                corner = surface.pickPoint(dem_origin, dem_target)
                if corner:
                    corner = Metashape.OrthoProjection.transform(corner, surface.projection, chunk_crs)
                    corner = T.inv().mulp(corner)
            else:
                corner = surface.pickPoint(ray_origin, ray_target)
            if not corner and chunk.tie_points:
                corner = chunk.tie_points.pickPoint(ray_origin, ray_target)
            if not corner:
                break
            corner = chunk.crs.project(T.mulp(corner))
            corners.append(corner)

        if len(corners) == 4:
            shape = chunk.shapes.addShape()
            shape.label = camera.label
            shape.attributes["Photo"] = camera.label
            shape.group = footprints
            shape.geometry = Metashape.Geometry.Polygon(corners)
        else:
            print("Skipping camera " + camera.label)

    with concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count()) as executor:
        executor.map(lambda camera: process_camera(chunk, camera), chunk.cameras)

    Metashape.app.update()
    print("Script finished!")

    doc.save()


project_psx_path = "/data/ATM/data_1/sfm/agi_projects/morrison_glacier/morrison_glacier.psx"
doc = Metashape.Document()
doc.open(project_psx_path)
create_footprints(doc)
