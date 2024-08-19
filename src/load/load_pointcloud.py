import subprocess
import numpy as np
import json
from plyfile import PlyData


def load_point_cloud(pointcloud_path):

    if pointcloud_path.endswith(".ply"):

        ply_data = PlyData.read(pointcloud_path)

        # Access the vertex data
        vertex_data = ply_data['vertex'].data

        # Extract x, y, z coordinates
        points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

        # Extract confidence values
        if 'confidence' in vertex_data.dtype.names:
            confidences = np.array(vertex_data['confidence']).reshape(-1, 1)
        else:
            confidences = None

        if confidences is not None:
            result_array = np.hstack((points, confidences))
        else:
            result_array = points

    else:

        conda_env_python = "/home/fdahle/miniconda3/envs/point_env/bin/python"
        script_path = "/src/load/o3d_wrapper.py"

        result = subprocess.run(
            [conda_env_python, script_path, pointcloud_path],
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )

        # parse the result
        result_array = np.array(json.loads(result.stdout))

    return result_array
