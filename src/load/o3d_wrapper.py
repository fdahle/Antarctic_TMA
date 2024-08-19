"""load a point cloud from a file"""

# Library imports
import sys
import json
import open3d as o3d
import numpy as np
from tqdm import tqdm


def load_pointcloud(pc_path: str, return_as_array=True) -> np.ndarray:
    """
    Loads a point cloud from a file and optionally returns it as a numpy array.

    Args:
        pc_path (str): The path to the point cloud file.
        return_as_array (bool, optional): Whether to return the point cloud as a numpy array. Defaults to False.

    Returns:
        np.ndarray: The loaded point cloud. If return_as_array is True, it returns a numpy array of shape (N, 3).
                    Otherwise, it returns an open3d.geometry.PointCloud object.
    """

    # extract the format of the point cloud
    pc_format = pc_path.split('.')[-1]

    # different formats require different loading methods
    if pc_format == "obj":

        points = []
        with open(pc_path, 'r') as file:
            for line in tqdm(file):
                if line.startswith('v '):  # Vertex description
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])

        # Create a point cloud object
        point_cloud = o3d.geometry.PointCloud()  # noqa
        point_cloud.points = o3d.utility.Vector3dVector(points)  # noqa

    else:
        point_cloud = o3d.io.read_point_cloud(pc_path)  # noqa

    # convert the points to numpy array
    if return_as_array:
        point_cloud = np.asarray(point_cloud.points)

    return point_cloud


if __name__ == "__main__":
    point_cloud_path = sys.argv[1]
    result = load_pointcloud(point_cloud_path)

    # Convert the result to a list and then to JSON string to return it
    result_list = result.tolist()
    print(json.dumps(result_list))
