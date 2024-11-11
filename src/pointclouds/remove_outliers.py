"""remove outliers from a point cloud"""
import numpy as np
import open3d as o3d
import time

from sklearn.decomposition import PCA


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

    # get points from the pc
    if isinstance(point_cloud, o3d.geometry.PointCloud):
        points = np.asarray(point_cloud.points)
    elif isinstance(point_cloud, np.ndarray):
        points = point_cloud[:, :3]
    else:
        raise ValueError("Input point cloud must be an Open3D PointCloud or a numpy array.")

    start = time.time()

    # Outlier removal using Statistical Outlier Removal
    pca = PCA(n_components=3)
    pca.fit(points)
    plane_normal = pca.components_[2]  # The normal vector of the main plane
    plane_point = pca.mean_  # A point on the plane (mean of points)

    # Calculate distances of each point to the plane
    distances = np.abs((points - plane_point).dot(plane_normal)) / np.linalg.norm(plane_normal)

    # Find statistical outliers based on distance from the plane
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Define a threshold for outliers (e.g., 3 standard deviations)
    threshold = mean_distance + 3 * std_distance

    # Separate inliers
    inlier_indices = np.where(distances <= threshold)[0]

    # print number of removed points
    print(f"Removed {len(points) - len(inlier_indices)} outliers from the point cloud")

    # select inliers from the original point cloud
    point_cloud_inliers = point_cloud.select_by_index(inlier_indices)

    end = time.time()
    print(f"Time taken to remove outliers: {end - start:.2f} seconds")

    # convert to numpy array if input was numpy array
    #if isinstance(point_cloud, np.ndarray):
        #print(ind)

    return point_cloud_inliers
