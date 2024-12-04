"""extract camera parameters from a Metashape chunk"""

import pandas as pd
import Metashape

def extract_camera_params(chunk: Metashape.Chunk,
                          num_digits: int = 6):
    """
    Extracts camera parameters from a Metashape chunk and returns them as a pandas DataFrame.
    The extracted parameters include calibration values, estimated and real camera locations,
    orientations, sensor dimensions, and transformation matrix elements.
    Args:
        chunk (Metashape.Chunk): The Metashape chunk object containing the cameras and transformations.
        num_digits (int, optional): Number of decimal places to round the calibration and coordinate values.
            Defaults to 6.
    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a camera and columns include:
            - label: Camera label.
            - focal_length: Sensor focal length.
            - calibration parameters (f, k1, k2, k3, k4, cx, cy, p1, p2, b1, b2).
            - camera center coordinates (x, y, z).
            - real and estimated camera locations and rotations.
            - sensor dimensions (width, height).
            - transformation matrix elements.
    """

    # Initialize list to store camera data
    camera_data = []

    # Chunk transformation matrix
    t_mat = chunk.transform.matrix

    # Loop through each camera in the chunk
    for camera in chunk.cameras:

        # Extract the 4x4 transform matrix if it exists
        if camera.transform:

            # compute the estimated location of the camera
            estimated_location = chunk.crs.project(chunk.transform.matrix.mulp(camera.center))

            # transformation matrix to the LSE coordinates in the given point
            m = chunk.crs.localframe(t_mat.mulp(camera.center))
            r = m * t_mat * camera.transform * Metashape.Matrix().Diag([1, -1, -1, 1])  # noqa

            # Normalize the rotation matrix
            row = [r.row(j) for j in range(3)]
            for j in range(3):
                row[j].size = 3
                row[j].normalize()
            r = Metashape.Matrix(row)  # noqa

            # Extract estimated rotation angles
            estimated_rotations = Metashape.utils.mat2ypr(r)

            # Extract 4x4 transformation matrix
            transform_matrix = camera.transform
            transform_params = [transform_matrix[i, j] for i in range(4) for j in range(4)]  # noqa

        else:
            # default values for missing transformation
            estimated_location = [None] * 3
            estimated_rotations = [None] * 3
            transform_params = [None] * 16

        # Append camera data to the list
        camera_data.append({
            "label": camera.label,
            "focal_length": camera.sensor.focal_length,
            "calibration_f": round(camera.calibration.f, 6),
            "calibration_k1": round(camera.calibration.k1, num_digits),
            "calibration_k2": round(camera.calibration.k2, num_digits),
            "calibration_k3": round(camera.calibration.k3, num_digits),
            "calibration_k4": round(camera.calibration.k4, num_digits),
            "calibration_cx": round(camera.calibration.cx, num_digits),
            "calibration_cy": round(camera.calibration.cy, num_digits),
            "calibration_p1": round(camera.calibration.p1, num_digits),
            "calibration_p2": round(camera.calibration.p2, num_digits),
            "calibration_b1": round(camera.calibration.b1, num_digits),
            "calibration_b2": round(camera.calibration.b2, num_digits),
            "center_x": round(camera.center.x, num_digits) if camera.center else None,
            "center_y": round(camera.center.y, num_digits) if camera.center else None,
            "center_z": round(camera.center.z, num_digits) if camera.center else None,
            "location_x_real": camera.reference.location.x if camera.reference.location else None,
            "location_y_real": camera.reference.location.y if camera.reference.location else None,
            "location_z_real": camera.reference.location.z if camera.reference.location else None,
            "location_x_estimated": estimated_location[0],
            "location_y_estimated": estimated_location[1],
            "location_z_estimated": estimated_location[2],
            "rotation_yaw_real": str(camera.reference.rotation) if camera.reference.rotation else None,
            "rotation_pitch_real": str(camera.reference.rotation) if camera.reference.rotation else None,
            "rotation_roll_real": str(camera.reference.rotation) if camera.reference.rotation else None,
            "rotation_yaw_estimated": estimated_rotations[0],
            "rotation_pitch_estimated": estimated_rotations[1],
            "rotation_roll_estimated": estimated_rotations[2],
            "sensor_width": camera.sensor.width if camera.sensor else None,
            "sensor_height": camera.sensor.height if camera.sensor else None,
            **{f"transform_{i + 1}": param for i, param in enumerate(transform_params)},  # Add transform parameters
        })

    camera_data = pd.DataFrame(camera_data)

    return camera_data