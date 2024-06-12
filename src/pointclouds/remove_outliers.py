"""remove outliers from a point cloud"""

# Library imports
import open3d as o3d

# debug variables
debug_show_pointcloud = False


def remove_outliers(point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:  # noqa
    """
    Removes outliers from a point cloud using Statistical Outlier Removal.

    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud to clean.

    Returns:
        o3d.geometry.PointCloud: The cleaned point cloud with outliers removed.
    """

    # Outlier removal using Statistical Outlier Removal
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Selecting the inlier points from the point cloud
    inlier_cloud = point_cloud.select_by_index(ind)

    # Visualize the cleaned point cloud (optional)
    print("Showing cleaned point cloud")
    o3d.visualization.draw_geometries([inlier_cloud])  # noqa

    return inlier_cloud
