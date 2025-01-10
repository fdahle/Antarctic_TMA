import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

# Debug variables
debug_show_pointcloud = False

def remove_outliers(point_cloud: o3d.geometry.PointCloud | np.ndarray) \
        -> o3d.geometry.PointCloud | np.ndarray:
    """
    Removes outliers from a point cloud using Statistical Outlier Removal.

    Args:
        point_cloud (o3d.geometry.PointCloud | np.ndarray): The input point cloud to clean.

    Returns:
        o3d.geometry.PointCloud | np.ndarray: The cleaned point cloud with outliers removed.
    """

    # Create an Open3D point cloud from the numpy array if necessary
    if isinstance(point_cloud, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    else:
        pcd = point_cloud

    # Outlier removal using PCA and distance thresholding
    pca = PCA(n_components=3, svd_solver='full')
    points = np.asarray(pcd.points)  # Extract points as numpy array for PCA
    pca.fit(points)
    plane_normal = pca.components_[2]  # Normal vector of the main plane
    plane_point = pca.mean_  # A point on the plane (mean of points)

    # Calculate distances of each point to the plane
    distances = np.abs((points - plane_point).dot(plane_normal)) / np.linalg.norm(plane_normal)
    distances = np.round(distances, decimals=8)

    # Find statistical outliers based on distance from the plane
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    threshold = mean_distance + 3 * std_distance  # Threshold for outliers

    # Identify inliers
    inlier_indices = np.where(distances <= threshold)[0]
    outlier_indices = np.where(distances > threshold)[0]

    # Print number of removed points
    print(f"Removed {len(points) - len(inlier_indices)} outliers from the point cloud")

    # Show the point cloud with inliers marked as red
    if debug_show_pointcloud:
        # Create a copy for inliers and outliers
        pcd_inliers = pcd.select_by_index(inlier_indices)
        pcd_outliers = pcd.select_by_index(outlier_indices)

        # Color inliers (default or green) and outliers (red)
        pcd_outliers.paint_uniform_color([1, 0, 0])  # Red for outliers

        # Display both together
        o3d.visualization.draw_geometries([pcd_inliers, pcd_outliers],
                                          window_name="Inliers (Green) and Outliers (Red)")

    # Select inliers based on input type
    if isinstance(point_cloud, np.ndarray):
        # Return a numpy array with only the inliers
        return point_cloud[inlier_indices, :]
    else:
        # Return an Open3D point cloud with only the inliers
        return pcd.select_by_index(inlier_indices)
