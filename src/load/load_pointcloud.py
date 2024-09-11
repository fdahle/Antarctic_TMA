import subprocess
import numpy as np
import json
from plyfile import PlyData


def load_point_cloud(pointcloud_path):
    """
    Load point cloud data from a file. Supports both PLY files and other formats via a wrapper script.

    Parameters:
    - pointcloud_path: str, the path to the point cloud file.

    Returns:
    - result_array: np.ndarray, the point cloud data as a numpy array.
    """

    if pointcloud_path.endswith(".ply"):
        ply_data = PlyData.read(pointcloud_path)

        # Access the vertex data
        vertex_data = ply_data['vertex'].data

        # Dynamically extract all available properties in the vertex data
        properties = vertex_data.dtype.names
        extracted_data = [vertex_data[prop] for prop in properties]

        # Stack the extracted data column-wise
        result_array = np.column_stack(extracted_data)

    else:
        conda_env_python = "/home/fdahle/miniconda3/envs/point_env/bin/python"
        script_path = "/src/load/o3d_wrapper.py"

        result = subprocess.run(
            [conda_env_python, script_path, pointcloud_path],
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )

        # Parse the result
        result_array = np.array(json.loads(result.stdout))

    return result_array
