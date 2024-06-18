"""reduce a point cloud by downsampling"""

import open3d as o3d
def reduce_pointcloud(point_cloud: o3d.geometry.PointCloud, reduce_type: str = "voxel",  # noqa
                      voxel_size: float = 10.0) -> o3d.geometry.PointCloud:  # noqa
    """
    Reduces the size of a point cloud using the specified reduction method.

    Args:
        point_cloud (geometry.PointCloud): The input point cloud to be reduced.
        reduce_type (str): The type of reduction method to use. Currently, it supports "voxel" only.
        voxel_size (float): The voxel size to use for voxel downsampling. Default is 10.0.

    Returns:
        geometry.PointCloud: The downsampled point cloud.

    Raises:
        ValueError: If an unknown reduction type is provided.
    """

    # Perform voxel downsampling
    if reduce_type == "voxel":
        downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    else:
        raise ValueError(f"Unknown point cloud reduction type: {reduce_type}")

    return downsampled
