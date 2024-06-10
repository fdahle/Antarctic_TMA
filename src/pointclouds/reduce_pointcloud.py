def reduce_pointcloud(point_cloud, reduce_type="voxel", voxel_size=10):

    # Perform voxel downsampling
    if reduce_type == "voxel":
        downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    else:
        raise ValueError(f"Unknown point cloud reduction type: {reduce_type}")

    return downsampled
