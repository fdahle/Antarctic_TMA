"""Export a point cloud as a ply file."""

import os.path
import open3d as o3d
import numpy as np
import pandas as pd

# noinspection PyUnresolvedReferences
def export_pointcloud(input_data: np.ndarray | o3d.t.geometry.PointCloud | pd.DataFrame,
                      output_path: str, write_ascii: bool = False,
                      overwrite: bool = False) -> None:
    """
    Export a point cloud as a ply file.

    Args:
        input_data (np.ndarray | o3d.geometry.PointCloud): Input data to be
            exported. This can be either a numpy array with shape (n, 3) where
            n is the number of points and each point has x, y, z coordinates,
            or an already created Open3D PointCloud object.
        output_path (str): The file path where the point cloud should be saved.
        write_ascii (bool): If True, the point cloud will be saved in ASCII format.
            Defaults to False.
        overwrite (bool): If True, the function will overwrite the file if it
            already exists. Defaults to False.

    Raises:
        FileExistsError: If the file already exists and `overwrite` is set
            to False.

    Returns:
        None: Saves the point cloud to a file in PLY format.
    """

    if os.path.isfile(output_path) and not overwrite:
        raise FileExistsError(f"'{output_path}' already exists.")

    if isinstance(input_data, pd.DataFrame):

        point_cloud = o3d.t.geometry.PointCloud()

        # Assign points if columns x, y, z are present
        if {'x', 'y', 'z'}.issubset(input_data.columns):
            points = input_data[['x', 'y', 'z']].to_numpy(dtype=np.float32)
            point_cloud.point['positions'] = o3d.core.Tensor(points)

        # Assign normals if columns nx, ny, nz are present
        if {'nx', 'ny', 'nz'}.issubset(input_data.columns):
            normals = input_data[['nx', 'ny', 'nz']].to_numpy(dtype=np.float32)
            point_cloud.point['normals'] = o3d.core.Tensor(normals)

        # Assign colors if columns red, green, blue are present
        if {'red', 'green', 'blue'}.issubset(input_data.columns):
            colors = input_data[['red', 'green', 'blue']].to_numpy(dtype=np.float32) / 255.0  # Normalize to [0, 1]
            point_cloud.point['colors'] = o3d.core.Tensor(colors)

        # assign custom attributes
        for col in input_data.columns:
            if col in {'x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue'}:
                continue

            custom_attribute = input_data[col].to_numpy(dtype=np.float32).reshape(-1, 1)
            point_cloud.point[col] = o3d.core.Tensor(custom_attribute)

    elif isinstance(input_data, np.ndarray):

        point_cloud = o3d.t.geometry.PointCloud()

        points = input_data[:, :3].astype(np.float32)
        point_cloud.point['positions'] = o3d.core.Tensor(points)

        # add optional normals
        if input_data.shape[1] > 3:
            normals = input_data[:, 6:9].astype(np.float32)
            point_cloud.point['normals'] = o3d.utility.Vector3dVector(normals)

        # add optional colors
        if input_data.shape[1] > 6:
            colors = input_data[:, 3:6].astype(np.float32)
            point_cloud.point['colors'] = o3d.utility.Vector3dVector(colors)

    # Check if the input data is already a PointCloud object
    elif isinstance(input_data, o3d.t.geometry.PointCloud):  # noqa
        point_cloud = input_data

    else:
        raise ValueError("Input data is not supported.")

    # Save the point cloud to a PLY file
    # noinspection PyUnresolvedReferences
    o3d.t.io.write_point_cloud(output_path, point_cloud, write_ascii=write_ascii)
