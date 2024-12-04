"""calc some stats for each image in an agisoft metashape project"""

# Local imports
import math

# Library imports
import Metashape


def get_image_stats(chunk):
    """
    nr of projections and error (px)
    Args:
        chunk:

    Returns:

    """

    # Access the tie points (sparse cloud points) in the chunk
    point_cloud = chunk.tie_points
    points = point_cloud.points

    if points is None:
        return {}, {}

    n_points = len(points)

    # Initialize lists and dictionaries for results
    point_ids = [-1] * len(point_cloud.tracks)  # Will map track_id to point index
    error_per_image = {}  # Store projection errors per image
    projections_per_image = {}  # Store the number of projections per image

    # Create a mapping from track_id to point index
    for point_id in range(n_points):
        point_ids[points[point_id].track_id] = point_id

    # iterate over all cameras
    for camera in chunk.cameras:

        # Skip Keyframe type cameras and cameras without a valid transformation matrix
        if camera.type == Metashape.Camera.Type.Keyframe or camera.transform is None:
            continue

        errors = []  # To accumulate error distances for this camera
        n_projections = 0  # Count valid projections for this camera
        point_index = 0  # Index to traverse points

        # Iterate over all projections of the current camera
        for proj in point_cloud.projections[camera]:

            # Get the track_id of the projection (linking it to a tie point)
            track_id = proj.track_id

            # Find the point with the same track_id
            while point_index < len(points) and points[point_index].track_id < track_id:
                point_index += 1

            # If point exists, check its validity and compute projection error
            if point_index < len(points) and points[point_index].track_id == track_id:
                # Skip invalid points
                if not points[point_index].valid:
                    continue

                # Calculate the projection error (squared distance between projected and actual point)
                dist = camera.error(points[point_index].coord, proj.coord).norm() ** 2
                errors.append(dist)

                # Increment valid projection count
                n_projections += 1

        # Store the square root of the mean error as the projection error for the camera
        if errors:
            error_per_image[camera.label] = round(math.sqrt(sum(errors) / len(errors)), 3)

        # Store the number of projections for the camera
        projections_per_image[camera.label] = n_projections

    return error_per_image, projections_per_image
