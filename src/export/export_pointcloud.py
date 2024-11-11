"""Export a point cloud as a ply file."""

import os.path
import open3d as o3d
import numpy as np
from typing import Union

overwrite = True


def export_pointcloud(input_data: Union[np.ndarray, o3d.data.PLYPointCloud],  # noqa
                      output_path: str, write_ascii=True) -> None:
    """
    Export a point cloud as a ply file.

    Args:
        input_data (Union[np.ndarray, o3d.geometry.PointCloud]): Input data to be exported.
            This can be either a numpy array with shape (n, 3) where n is the number of points
            and each point has x, y, z coordinates, or an already created Open3D PointCloud object.
        output_path (str): The file path where the point cloud should be saved.

    Raises:
        FileExistsError: If the file already exists and `overwrite` is set to False.

    Returns:
        None: Saves the point cloud to a file in PLY format.
    """

    if os.path.isfile(output_path) and not overwrite:
        raise FileExistsError(f"'{output_path}' already exists.")

    # Check if the input data is already a PointCloud object
    if isinstance(input_data, o3d.geometry.PointCloud):  # noqa
        point_cloud = input_data
    else:
        # Create a PointCloud object and set points
        point_cloud = o3d.geometry.PointCloud()  # noqa
        point_cloud.points = o3d.utility.Vector3dVector(input_data[:, :3])

        # add optional colors
        if point_cloud.shape[1] > 3:
            point_cloud.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6])

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(output_path, point_cloud, write_ascii=write_ascii)
