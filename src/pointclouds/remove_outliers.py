import open3d as o3d

debug_show_pointcloud = False

def remove_outliers(point_cloud):

    # Outlier removal using Statistical Outlier Removal
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Selecting the inlier points from the point cloud
    inlier_cloud = point_cloud.select_by_index(ind)

    # Visualize the cleaned point cloud (optional)
    print("Showing cleaned point cloud")
    o3d.visualization.draw_geometries([inlier_cloud])

    return inlier_cloud