import open3d as o3d
import numpy as np
from tqdm import tqdm

def load_pointcloud(pc_path: str, return_as_array=False) -> np.ndarray:

    # extract the format of the point cloud
    format = pc_path.split('.')[-1]

    # different formats require different loading methods
    if format == "obj":

        points = []
        with open(pc_path, 'r') as file:
            for line in tqdm(file):
                if line.startswith('v '):  # Vertex description
                    parts = line.strip().split()
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])

        # Create a point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

    else:
        point_cloud = o3d.io.read_point_cloud(pc_path)

    # convert the points to numpy array
    if return_as_array:
        point_cloud = np.asarray(point_cloud.points)

    return point_cloud
