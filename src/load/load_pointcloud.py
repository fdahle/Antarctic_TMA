import open3d as o3d
import numpy as np


def load_pointcloud(pc_path: str) -> np.ndarray:

    # load the point cloud
    point_cloud = o3d.io.read_point_cloud(pc_path)

    # convert the points to numpy array
    points = np.asarray(point_cloud.points)

    return points
