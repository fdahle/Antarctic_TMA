import itertools
import numpy as np

import src.display.display_images as di
import src.load.load_image as li


def save_tie_points(chunk, save_path=None):

    # get data from the chunk
    cameras = chunk.cameras
    point_cloud = chunk.tie_points

    # get data from the point cloud
    projections = point_cloud.projections
    points = point_cloud.points
    #tracks = point_cloud.tracks

    if points is None:
        raise Exception("No tie points available")
        return

    point_ids = [-1] * len(point_cloud.tracks)
    for point_id in range(0, len(points)):
        point_ids[points[point_id].track_id] = point_id
    camera_matches_valid = dict()

    # iterate over all cameras
    for camera1, camera2 in itertools.combinations(cameras, 2):

        print(f"Camera pair: {camera1.label}, {camera2.label}")

        # Check if camera transforms are available
        if camera1.transform is None or camera2.transform is None:
            print("Skipping pair due to missing transforms")
            continue

        # Get valid matches for each camera
        for camera in [camera1, camera2]:
            valid_matches = set()
            for proj in projections[camera]:
                track_id = proj.track_id
                point_id = point_ids[track_id]

                if point_id < 0:
                    continue
                if not points[point_id].valid:  # skipping invalid points
                    continue
                valid_matches.add(point_id)
            camera_matches_valid[camera] = valid_matches

        # Get valid matches
        valid = camera_matches_valid[camera1].intersection(camera_matches_valid[camera2])

        # Get total number of matches
        total = set([p.track_id for p in projections[camera1]]).intersection(
            set([p.track_id for p in projections[camera2]]))

        print(f"Valid/Total matches: {len(valid), len(total)}")

        # Prepare numpy arrays for valid and invalid matches
        valid_matches_array = []
        invalid_matches_array = []

        # Collect coordinates for valid matches
        for track_id in valid:
            try:
                proj1 = next(p for p in projections[camera1] if p.track_id == track_id)
                proj2 = next(p for p in projections[camera2] if p.track_id == track_id)
                valid_matches_array.append([proj1.coord.x, proj1.coord.y, proj2.coord.x, proj2.coord.y])
            except StopIteration:
                continue

        # Collect coordinates for invalid matches
        for track_id in total - valid:
            try:
                proj1 = next(p for p in projections[camera1] if p.track_id == track_id)
                proj2 = next(p for p in projections[camera2] if p.track_id == track_id)
                invalid_matches_array.append([proj1.coord.x, proj1.coord.y, proj2.coord.x, proj2.coord.y])
            except StopIteration:
                continue

        # Convert to numpy arrays
        valid_matches_array = np.array(valid_matches_array)
        invalid_matches_array = np.array(invalid_matches_array)

        # check if there are any matches
        if valid_matches_array.shape[0] == 0 and invalid_matches_array.shape[0] == 0:
            continue
        elif valid_matches_array.shape[0] == 0:
            matches_array = invalid_matches_array
        elif invalid_matches_array.shape[0] == 0:
            matches_array = valid_matches_array
        else:
            matches_array = np.vstack((valid_matches_array, invalid_matches_array))

        # Create conf list with 1 for valid matches and 0 for invalid matches
        valid_conf = [1] * len(valid_matches_array)
        invalid_conf = [0] * len(invalid_matches_array)
        conf = valid_conf + invalid_conf

        # Load images
        image1 = li.load_image(camera1.label)
        image2 = li.load_image(camera2.label)

        if save_path is not None:
            image_path = f"{save_path}/{camera1.label}_{camera2.label}.png"
        else:
            image_path = None

        # Display tie points
        di.display_images([image1, image2], tie_points=matches_array,
                          tie_points_conf=conf, save_path=image_path)