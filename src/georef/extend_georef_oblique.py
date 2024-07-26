# extend the geo-referencing to the oblique images

import os
import geopandas as gpd
import math
import numpy as np
import shapely
from shapely import geometry
from vector3d.vector import Vector

import src.base.connect_to_database as ctd

# constants
FEET_TO_METER = 0.3048
ADAPT_POLY_WITH_REMA = False


def extend_georef_oblique(point_data, georef_type, output_fld, conn=None):
    # create connection to the database if not provided
    if conn is None:
        conn = ctd.establish_connection()

    # Initialize a list to store the new polygons
    polygons = []

    # iterate all rows in the geopandas dataframe
    for _, row in point_data.iterrows():

        # get the values from the row
        image_id = row["image_id"]
        position = row["position_exact"]
        altitude = row["height"]
        azimuth = row["azimuth"]

        if position is None:
            print("No position found for image_id: ", image_id)
            continue
        if altitude is None or np.isnan(altitude):
            print("No altitude found for image_id: ", image_id)
            continue
        if azimuth is None or np.isnan(azimuth):
            print("No azimuth found for image_id: ", image_id)
            continue

        # get the right ids
        image_id_left = image_id.replace('32V', '31L')
        image_id_right = image_id.replace('32V', '33R')

        # get x and y
        point = shapely.from_wkt(position)
        x = point.x
        y = point.y

        # Convert altitude from feet to meters
        altitude_meters = altitude * FEET_TO_METER

        # get focal length
        try:
            sql_string_left = f"SELECT focal_length FROM images_extracted WHERE image_id = '{image_id_left}'"
            focal_length_left = ctd.execute_sql(sql_string_left, conn).iloc[0]["focal_length"]
        except (Exception,):
            focal_length_left = None
        try:
            sql_string_right = f"SELECT focal_length FROM images_extracted WHERE image_id = '{image_id_right}'"
            focal_length_right = ctd.execute_sql(sql_string_right, conn).iloc[0]["focal_length"]
        except (Exception,):
            focal_length_right = None

        # Calculate the footprints for left and right cameras
        if focal_length_left is None or np.isnan(focal_length_left):
            print("No focal length found for image_id: ", image_id_left)
        else:
            left_footprint = _calc_footprint(x, y, azimuth, altitude_meters,
                                             'L', focal_length_left)
            # Add the polygons to the list
            polygons.append({
                "image_id": image_id_left,
                "geometry": left_footprint
            })

        if focal_length_right is None or np.isnan(focal_length_right):
            print("No focal length found for image_id: ", image_id_right)
        else:
            right_footprint = _calc_footprint(x, y, azimuth, altitude_meters,
                                              'R', focal_length_right)
            polygons.append({
                "image_id": image_id_right,
                "geometry": right_footprint
            })

    # Create a GeoDataFrame from the polygons
    gdf = gpd.GeoDataFrame(polygons, columns=["image_id", "geometry"], crs="EPSG:3031")

    # Save the GeoDataFrame to a file
    output_file = os.path.join(output_fld, f"{georef_type}_oblique.shp")
    gdf.to_file(output_file)

    print(f"Geo-referenced oblique footprints saved to {output_file}")


def _calc_footprint(x, y, azimuth, altitude,
                    view_direction, focal_length):
    """
    "https://gis.stackexchange.com/questions/75405/aerial-photograph-footprint-size-calculation"

    Args:
        img_id:
        x:
        y:
        azimuth:
        altitude:
        view_direction:
        focal_length:

    Returns:

    """

    if view_direction == "L":
        gamma = 30
    elif view_direction == "R":
        gamma = 330
    else:
        raise ValueError("view_direction must be either 'L' or 'R'")

    print("T")

    # here the camera params are saved
    camera_params = {
        "alpha": azimuth,
        "beta": 0,
        "gamma": gamma,
        # "fovV": 2 * math.atan(height_in_mm / (2 * focal_length)),
        # "fovH": 2 * math.atan(width_in_mm / (2 * focal_length)),
        "fovV": 60,
        "fovH": 60,
        "xPos": x,
        "yPos": y,
        "zPos": altitude,
        "fx": focal_length / 1000,  # / 1000,  # convert to m
        "fy": focal_length / 1000,  # / 1000,
        "px": 0.009,  # / 1000,
        "py": -0.001  # / 1000,
    }

    # actual function to get the footprint
    def get_bounds(cam_params):
        # convert to radians
        alpha_r = math.radians(cam_params["alpha"])
        beta_r = math.radians(cam_params["beta"])
        gamma_r = math.radians(cam_params["gamma"])

        # calculate intrinsic matrix
        # intrinsics = np.array([[cam_params["fx"], 0, cam_params["px"]],
        #                       [0, cam_params["fy"], cam_params["py"]],
        #                       [0, 0, 1]])

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

        # create camera calculator
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

            def __init__(self):
                pass

            def __del__(self):
                pass

            @staticmethod
            def get_bounding_polygon(fov_h, fov_v, _altitude, roll, pitch, heading):
                """
                Get corners of the polygon captured by the camera on the ground.
                The calculations are performed in the axes origin (0, 0, altitude)
                and the points are not yet translated to camera's X-Y coordinates.
                Parameters:
                    fov_h (float): Horizontal field of view in radians
                    fov_v (float): Vertical field of view in radians
                    _altitude (float): Altitude of the camera in meters
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

                origin_vec = Vector(0, 0, _altitude)
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
                    _origin (vector3d.vector.Vector): Position of the camera. The computation were
                        developed assuming the camera was at the axes origin (0, 0, altitude) and the
                        results translated by the camera's real position afterward.
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
            points.append([p.x + camera_params["xPos"], p.y + camera_params["yPos"]])

        # get center of list of points
        all_x = [p[0] for p in points]
        all_y = [p[1] for p in points]
        ori = (sum(all_x) / len(points), sum(all_y) / len(points))
        ref_vector = [0, 1]

        # function to sort points clockwise
        def clockwise_angle_and_distance(point):
            # Vector between point and the origin: v = p - o
            vector = [point[0] - ori[0], point[1] - ori[1]]
            # Length of vector: ||v||
            len_vector = math.hypot(vector[0], vector[1])
            # If length is zero there is no angle
            if len_vector == 0:
                return -math.pi, 0
            # Normalize vector: v/||v||
            normalized = [vector[0] / len_vector, vector[1] / len_vector]
            dot_product = normalized[0] * ref_vector[0] + normalized[1] * ref_vector[1]  # x1*x2 + y1*y2
            diff_product = ref_vector[1] * normalized[0] - ref_vector[0] * normalized[1]  # x1*y2 - y1*x2
            angle = math.atan2(diff_product, dot_product)
            # Negative angles represent counter-clockwise angles, so we need to subtract them
            # from 2*pi (360 degrees)
            if angle < 0:
                return 2 * math.pi + angle, len_vector
            # I return first the angle because that's the primary sorting criterium
            # but if two vectors have the same angle then the shorter distance should come first.
            return angle, len_vector

        # sort the points
        points = sorted(points, key=clockwise_angle_and_distance)

        # create polygon from points
        poly = geometry.Polygon(points)

        return poly, ori

    # get initial footprint
    polygon, origin = get_bounds(camera_params)

    if ADAPT_POLY_WITH_REMA:
        # get average elevation data for this initial footprint
        rema_data = grd.get_rema_data(polygon.bounds)
        avg_ground_height = np.average(rema_data)

        # recalculate the height of camera (in relation to the ground)
        altitude = altitude - avg_ground_height
        camera_params["zPos"] = altitude

        # calculate polygon again
        polygon, origin = get_bounds(camera_params)

    return polygon
