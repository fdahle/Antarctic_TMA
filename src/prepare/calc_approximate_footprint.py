"""Calculates the approximate footprint of an image"""

# Library imports
import math
import numpy as np
from shapely import geometry
from typing import Union
from vector3d.vector import Vector

# Local imports
import src.load.load_rema as lr


def calc_approximate_footprint(center: Union[Vector, tuple[float, float]], azimuth: float,
                               view_direction: str, altitude: Union[int, float],
                               focal_length: Union[int, float],
                               adapt_with_rema: bool = False) -> geometry.Polygon:

    """
    Calculates the approximate footprint of an image based on the camera parameters like center,
    azimuth & view_direction. Note that the ground elevation data is not considered in this calculation.
    However, it is possible to adjust the footprint based on REMA elevation data.
    Args:
        center: The center point of the camera's view. This can be a shapely.geometry.Point
                or a tuple of floats representing the X and Y coordinates.
        azimuth: The azimuth angle of the camera, in degrees.
        view_direction: The direction the camera is facing, represented as a single character.
                        'L' for left, 'V' for vertical, and 'R' for right.
        altitude: The altitude of the camera, in feet.
        focal_length: The focal length of the camera, in millimeters.
        adapt_with_rema: A boolean indicating whether to adjust the calculation based on REMA data.
    Returns:
        A shapely.geometry.Polygon representing the approximate footprint of the camera's view.
    Raises:
        ValueError: If `view_direction` is not one of the expected values,
                    if `altitude` is None, or if `focal_length` is None.
    """

    if view_direction not in ["L", "V", "R"]:
        raise ValueError("View direction must be either 'L' (left), 'V' (vertical), or 'R' (right)")
    if altitude is None:
        raise ValueError("Altitude is required to calculate the approximate footprint")
    if focal_length is None:
        raise ValueError("Focal length is required to calculate the approximate footprint")

    # get gamma
    gamma = 0
    if view_direction == "L":
        gamma = 30
    elif view_direction == "R":
        gamma = 330

    # convert center to shapely point if it's a tuple
    if isinstance(center, tuple):
        center = geometry.Point(center)

    # create camera matrix
    camera_params = {
        # "alpha": 90 + azimuth,
        "alpha": 360 - azimuth + 180,
        "beta": 0,
        "gamma": gamma,
        "fovV": 60,
        "fovH": 60,
        "xPos": center.x,
        "yPos": center.y,
        "zPos": altitude,
        "fx": focal_length / 1000,  # / 1000,  # convert to m
        "fy": focal_length / 1000,  # / 1000,
        "px": 0.009,  # / 1000,
        "py": -0.001  # / 1000,
    }

    # convert altitude from feet to meter & save
    altitude = int(altitude / 3.281)
    camera_params["zPos"] = altitude

    # calculate an initial approx_footprint based on camera_params
    polygon = _get_bounds(camera_params)

    # if true a new polygon is calculated based on the average elevation data from rema
    if adapt_with_rema:

        # get average elevation data for this initial approx_footprint based on rema data
        rema_data = lr.load_rema(polygon, zoom_level=32)
        avg_ground_height = np.average(rema_data)

        # convert to feet
        avg_ground_height = avg_ground_height * 3.281

        # recalculate the height of camera (in relation to the ground)
        altitude = altitude - avg_ground_height
        camera_params["zPos"] = altitude

        # recalculate the approx_footprint based on the new camera_params
        polygon = _get_bounds(camera_params)

    return polygon


def _get_bounds(cam_params: dict) -> geometry.Polygon:
    """
    Calculates a bounding polygon based on camera parameters.
    Args:
        cam_params: A dictionary containing camera parameters including
                    alpha (rotation angle), beta, gamma (viewing angles),
                    fovV (vertical field of view), fovH (horizontal field of view),
                    xPos, yPos, zPos (camera position), fx, fy (focal lengths),
                    px, py (pixel size).
    Returns:
        A shapely.geometry.Polygon representing the calculated bounding area.
    """

    # convert to radians
    alpha_r = math.radians(cam_params["alpha"])
    beta_r = math.radians(cam_params["beta"])
    gamma_r = math.radians(cam_params["gamma"])

    # calculate rotation matrix
    rot_z = np.array([[np.cos(alpha_r), -np.sin(alpha_r), 0],
                      [np.sin(alpha_r), np.cos(alpha_r), 0],
                      [0, 0, 1]])

    rot_y = np.array([[np.cos(beta_r), 0, np.sin(beta_r)],
                      [0, 1, 0],
                      [-np.sin(beta_r), 0, np.cos(beta_r)]])

    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(gamma_r), -np.sin(gamma_r)],
                      [0, np.sin(gamma_r), np.cos(gamma_r)]])

    rotation = rot_z.dot(rot_y).dot(rot_x)

    # calculate translation vector
    translation = np.array([cam_params["xPos"],
                            cam_params["yPos"],
                            cam_params["zPos"]])

    # calculate cameraPose
    _r1 = rotation[:, 0]
    _r2 = rotation[:, 1]
    _temp = np.vstack((_r1, _r2, translation)).T

    # init the camera calculator
    cc = CameraCalculator()

    # create bounding box
    bbox = cc.get_bounding_polygon(
        math.radians(cam_params["fovH"]),
        math.radians(cam_params["fovV"]),
        cam_params["zPos"],
        math.radians(cam_params["gamma"]),  # roll: 0 for V; 30 for L, 360 - 30 for R
        math.radians(cam_params["beta"]),  # pitch
        math.radians(cam_params["alpha"]))  # heading

    # convert bbox to absolute coordinate points
    points = []
    for p in bbox:  # noqa
        points.append((p.x + cam_params["xPos"], p.y + cam_params["yPos"]))

    # sort points
    points = _sort_points(points)

    # create polygon from points
    polygon = geometry.Polygon(points)

    return polygon


def _sort_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Sorts a list of points in 2D space in a counter-clockwise direction starting from the upward direction,
    relative to the centroid of all points. If two points form the same angle with the centroid,
    the one closer to the centroid comes first.
    Args:
        points: A list of points (tuples) where each point is represented as (x, y).
    Returns:
        The sorted list of points based on their angle and distance relative to the centroid.
    """

    # Calculate the centroid of the points
    all_x = [po[0] for po in points]
    all_y = [po[1] for po in points]
    centroid = (sum(all_x) / len(points), sum(all_y) / len(points))

    # Function to calculate angle and distance of a point from the centroid
    def __calc_angle_and_distance(point: tuple[float, float]) -> tuple[float, float]:

        # Vector from point to centroid
        vector = [point[0] - centroid[0], point[1] - centroid[1]]
        len_vector = math.hypot(vector[0], vector[1])  # Length of vector

        # Avoid division by zero
        if len_vector == 0:
            return -math.pi, 0

        # Normalize vector
        normalized = [vector[0] / len_vector, vector[1] / len_vector]
        # Reference vector for upward direction
        ref_vector = [0, 1]

        # Dot product and determinant (for angle calculation)
        dot_product = normalized[0] * ref_vector[0] + normalized[1] * ref_vector[1]
        diff_product = ref_vector[1] * normalized[0] - ref_vector[0] * normalized[1]
        angle = math.atan2(diff_product, dot_product)

        # Adjust angles to ensure counter-clockwise sorting
        if angle < 0:
            angle = 2 * math.pi + angle

        return angle, len_vector

    # Sort points based on angle and distance from the centroid
    sorted_points = sorted(points, key=__calc_angle_and_distance)

    return sorted_points


class CameraCalculator:
    """Porting of CameraCalculator.java

    This code is a 1to1 python porting of the java code:
        https://github.com/zelenmi6/thesis/blob/master/src/geometry/CameraCalculator.java
    referred in:
        https://stackoverflow.com/questions/38099915/calculating-coordinates-of-an-oblique-aerial-image
    The only part not ported are that explicitly abandoned or not used at all by the main
    call to get_bounding_polygon method.
    by: milan zelenka
    https://github.com/zelenmi6
    https://stackoverflow.com/users/6528363/milan-zelenka

    example:

        c=CameraCalculator()
        bbox=c.get_bounding_polygon(
            math.radians(62),
            math.radians(84),
            117.1,
            math.radians(0),
            math.radians(33.6),
            math.radians(39.1))
        for i, p in enumerate(bbox):
            print("point:", i, '-', p.x, p.y, p.z)
    """

    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        pass

    @staticmethod
    def get_bounding_polygon(fov_h, fov_v, alti, roll, pitch, heading):
        """
        Get corners of the polygon captured by the camera on the ground.
        The calculations are performed in the axes origin (0, 0, altitude)
        and the points are not yet translated to camera's X-Y coordinates.
        Parameters:
            fov_h (float): Horizontal field of view in radians
            fov_v (float): Vertical field of view in radians
            alti (float): Altitude of the camera in meters
            heading (float): Heading of the camera (z-axis) in radians
            roll (float): Roll of the camera (x-axis) in radians
            pitch (float): Pitch of the camera (y-axis) in radians
        Returns:
            vector3d.vector.Vector: Array with 4 points defining a polygon
        """

        # import ipdb; ipdb.set_trace()
        ray11 = CameraCalculator.ray1(fov_h, fov_v)
        ray22 = CameraCalculator.ray2(fov_h, fov_v)
        ray33 = CameraCalculator.ray3(fov_h, fov_v)
        ray44 = CameraCalculator.ray4(fov_h, fov_v)

        rotated_vectors = CameraCalculator.rotate_rays(
            ray11, ray22, ray33, ray44, roll, pitch, heading)

        origin_vec = Vector(0, 0, alti)
        intersections = CameraCalculator.get_ray_ground_intersections(rotated_vectors, origin_vec)

        return intersections

    # Ray-vectors defining the camera's field of view. FOVh and FOVv are interchangeable
    # depending on the camera's orientation
    @staticmethod
    def ray1(fov_h, fov_v):
        """
        Parameters:
            fov_h (float): Horizontal field of view in radians
            fov_v (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        """
        ray = Vector(math.tan(fov_v / 2), math.tan(fov_h / 2), -1)
        return ray.normalize()

    @staticmethod
    def ray2(fov_h, fov_v):
        """
        Parameters:
            fov_h (float): Horizontal field of view in radians
            fov_v (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        """
        ray = Vector(math.tan(fov_v / 2), -math.tan(fov_h / 2), -1)
        return ray.normalize()

    @staticmethod
    def ray3(fov_h, fov_v):
        """
        Parameters:
            fov_h (float): Horizontal field of view in radians
            fov_v (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        """
        ray = Vector(-math.tan(fov_v / 2), -math.tan(fov_h / 2), -1)
        return ray.normalize()

    @staticmethod
    def ray4(fov_h, fov_v):
        """
        Parameters:
            fov_h (float): Horizontal field of view in radians
            fov_v (float): Vertical field of view in radians
        Returns:
            vector3d.vector.Vector: normalised vector
        """
        ray = Vector(-math.tan(fov_v / 2), math.tan(fov_h / 2), -1)
        return ray.normalize()

    @staticmethod
    def rotate_rays(ray1, ray2, ray3, ray4, roll, pitch, yaw):
        """Rotates the four ray-vectors around all 3 axes
        Parameters:
            ray1 (vector3d.vector.Vector): First ray-vector
            ray2 (vector3d.vector.Vector): Second ray-vector
            ray3 (vector3d.vector.Vector): Third ray-vector
            ray4 (vector3d.vector.Vector): Fourth ray-vector
            roll (float): Roll rotation
            pitch (float): Pitch rotation
            yaw (float): Yaw rotation
        Returns:
            Returns new rotated ray-vectors
        """
        sin_alpha = math.sin(yaw)
        sin_beta = math.sin(pitch)
        sin_gamma = math.sin(roll)
        cos_alpha = math.cos(yaw)
        cos_beta = math.cos(pitch)
        cos_gamma = math.cos(roll)
        m00 = cos_alpha * cos_beta
        m01 = cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma
        m02 = cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma
        m10 = sin_alpha * cos_beta
        m11 = sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma
        m12 = sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma
        m20 = -sin_beta
        m21 = cos_beta * sin_gamma
        m22 = cos_beta * cos_gamma

        # Matrix rotationMatrix = new Matrix(new double[][]{{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}})
        rotation_matrix = np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])

        ray1_matrix = np.array([[ray1.x], [ray1.y], [ray1.z]])
        ray2_matrix = np.array([[ray2.x], [ray2.y], [ray2.z]])
        ray3_matrix = np.array([[ray3.x], [ray3.y], [ray3.z]])
        ray4_matrix = np.array([[ray4.x], [ray4.y], [ray4.z]])

        res1 = rotation_matrix.dot(ray1_matrix)
        res2 = rotation_matrix.dot(ray2_matrix)
        res3 = rotation_matrix.dot(ray3_matrix)
        res4 = rotation_matrix.dot(ray4_matrix)

        rotated_ray1 = Vector(res1[0, 0], res1[1, 0], res1[2, 0])
        rotated_ray2 = Vector(res2[0, 0], res2[1, 0], res2[2, 0])
        rotated_ray3 = Vector(res3[0, 0], res3[1, 0], res3[2, 0])
        rotated_ray4 = Vector(res4[0, 0], res4[1, 0], res4[2, 0])
        ray_array = [rotated_ray1, rotated_ray2, rotated_ray3, rotated_ray4]

        return ray_array

    @staticmethod
    def get_ray_ground_intersections(rays, _origin):
        """
        Finds the intersections of the camera's ray-vectors
        and the ground approximated by a horizontal plane
        Parameters:
            rays (vector3d.vector.Vector[]): Array of 4 ray-vectors
            _origin (vector3d.vector.Vector): Position of the camera. The computation were developed
                                            assuming the camera was at the axes origin (0, 0, altitude) and the
                                            results translated by the camera's real position afterwards.
        Returns:
            vector3d.vector.Vector
        """
        # Vector3d [] intersections = new Vector3d[rays.length];
        # for (int i = 0; i < rays.length; i ++) {
        #     intersections[i] = CameraCalculator.find_ray_ground_intersection(rays[i], origin);
        # }
        # return intersections

        # 1to1 translation without python syntax optimisation
        intersections = []
        for i in range(len(rays)):
            intersections.append(CameraCalculator.find_ray_ground_intersection(rays[i], _origin))
        return intersections

    @staticmethod
    def find_ray_ground_intersection(ray, _origin):
        """
        Finds a ray-vector's intersection with the ground approximated by a plane
        Parameters:
            ray (vector3d.vector.Vector): Ray-vector
            _origin (vector3d.vector.Vector): Camera's position
        Returns:
            vector3d.vector.Vector
        """
        # Parametric form of an equation
        # P = origin + vector * t
        x_vec = Vector(_origin.x, ray.x)
        y_vec = Vector(_origin.y, ray.y)
        z_vec = Vector(_origin.z, ray.z)

        # Equation of the horizontal plane (ground)
        # -z_vec = 0

        # Calculate t by substituting z
        t = - (z_vec.x / z_vec.y)

        # Substitute t in the original parametric equations to get points of intersection
        return Vector(x_vec.x + x_vec.y * t, y_vec.x + y_vec.y * t, z_vec.x + z_vec.y * t)
