import Metashape
from numba.np.arraymath import ov_np_where

overwrite=True

def limit_area(doc, ground_elevation=0):

    chunk = doc.chunk

    if chunk is None:
        doc = Metashape.app.document
        chunk = doc.chunk
    if not chunk:
        print("No active chunk found.")
        return

    if not chunk.shapes:
        chunk.shapes = Metashape.Shapes()
        chunk.shapes.crs = chunk.crs

    footprints = chunk.shapes.addGroup()
    footprints.label = "Footprints"
    footprints.color = (30, 239, 30)

    for camera in chunk.cameras:
        # Skip cameras without a valid transform
        if not camera.transform:
            continue

        # Compute the footprint polygon for this camera
        footprint = get_camera_footprint(camera, ground_elevation)
        if footprint is None:
            continue

        # Create a polygon geometry from the footprint points
        poly = Metashape.Geometry.Polygon(footprint)

        # Add a new shape to the chunk and assign the polygon geometry and a label
        shape = chunk.shapes.addShape()
        shape.label = camera.label
        shape.attributes["Photo"] = camera.label
        shape.geometry = poly
        shape.group = footprints

        print("Added footprint for camera: {}".format(camera.label))

    print("Camera footprints creation complete.")
    doc.save()

def compute_plane_intersection(camera, uv, ground_elevation):
    """
    Fallback: compute intersection with a horizontal plane at ground_elevation.
    """
    ray = camera.unproject(uv)
    center = camera.transform.translation()  # camera center in world coordinates
    if abs(ray.z) < 1e-6:
        print("Ray nearly parallel to ground for camera {} at {}".format(camera.label, uv))
        return None
    t = (ground_elevation - center.z) / ray.z
    if t < 0:
        print("Intersection behind camera {} at {}".format(camera.label, uv))
        return None
    return center + ray * t


def get_camera_footprint(camera, chunk, ground_elevation=0.0, use_raycast=False, max_distance=1000, tolerance=1e-3):
    """
    Calculate the footprint of a camera by projecting its image corners onto a horizontal plane
    at the given ground elevation.

    Parameters:
        camera (Metashape.Camera): The camera for which to compute the footprint.
        ground_elevation (float): The elevation (Z value) at which to project the image corners.

    Returns:
        list of Metashape.Vector: A list of 3D points (as a polygon) representing the camera footprint.
    """
    if not camera.transform:
        print("Camera {} has no valid transform.".format(camera.label))
        return None

    # Retrieve sensor information (image dimensions in pixels)
    sensor = camera.sensor
    width = sensor.width
    height = sensor.height

    # Define the four image corners (in pixel coordinates)
    corners = [
        Metashape.Vector([0, 0]),
        Metashape.Vector([width, 0]),
        Metashape.Vector([width, height]),
        Metashape.Vector([0, height])
    ]

    footprint = []
    for uv in corners:
        world_point = None
        if use_raycast:
            # Attempt raycast. The returned tuple is assumed to be:
            # (hit, hit_point, distance, face) where hit is a boolean.
            hit = chunk.raycast(camera, uv, max_distance, tolerance)
            if hit[0]:
                world_point = hit[1]
            else:
                print("Raycast did not hit ground for camera {} at pixel {}. Using plane intersection.".format(camera.label, uv))
        # Fallback if raycast is disabled or did not hit:
        if world_point is None:
            world_point = compute_plane_intersection(camera, uv, ground_elevation)
            if world_point is None:
                print("Could not compute intersection for camera {} at pixel {}.".format(camera.label, uv))
                return None
        footprint.append(world_point)
    return footprint


if __name__ == "__main__":
    project_psx_path = "/data/ATM/data_1/sfm/agi_projects/morrison_glacier/morrison_glacier.psx"
    doc = Metashape.Document()
    doc.open(project_psx_path)
    limit_area(doc)
