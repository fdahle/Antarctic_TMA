import os

import numpy as np

import open3d as o3d

def load_point_cloud(pointcloud_path, return_arr=False):
    """
    Load point cloud data from a file. Supports both PLY files and other formats via a wrapper script.

    Parameters:
    - pointcloud_path: str, the path to the point cloud file.

    Returns:
    - result_array: np.ndarray, the point cloud x y z as a numpy array.
    """

    # check if the point cloud exists
    if not os.path.exists(pointcloud_path):
        raise FileNotFoundError(f"The point cloud file {pointcloud_path} "
                                f"does not exist.")

    pcd = o3d.io.read_point_cloud(pointcloud_path)

    # Convert the point cloud to a numpy array
    if return_arr:
        pcd = np.asarray(pcd.points)

    return pcd


if __name__ == "__main__":
    PATH_PROJECT_FOLDERS = "/data/ATM/data_1/sfm/agi_projects"
    project_name = ("grf_test")
    project_fld = os.path.join(PATH_PROJECT_FOLDERS, project_name)
    output_fld = os.path.join(project_fld, "output")
    output_path_pc_rel = os.path.join(output_fld,
                                      project_name + "_pointcloud_relative.ply")

    point_cloud = load_point_cloud(output_path_pc_rel)
    print(np.asarray(point_cloud).shape)
