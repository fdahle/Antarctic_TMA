def reduce_pointcloud(point_cloud, type="voxel", voxel_size=10):

    # Perform voxel downsampling
    downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    return downsampled
