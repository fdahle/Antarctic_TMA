import os.path

import numpy as np
import open3d as o3d

overwrite = False

def export_pointcloud(input_data, output_path: str):

    if os.path.isfile(output_path) and not overwrite:
        raise FileExistsError(f"'{output_path}' already exists.")

    # Check if the input data is already a PointCloud object
    if isinstance(input_data, o3d.geometry.PointCloud):
        point_cloud = input_data
    else:
        # Create a PointCloud object and set points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(input_data))

    # Save the point cloud to a PLY file
    o3d.io.write_point_cloud(output_path, point_cloud, write_ascii=True)