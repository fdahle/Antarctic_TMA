
# Library imports
import numpy as np
import Metashape
from tqdm import tqdm

debug_print_mode = False

def add_gcp_markers(chunk,
                    markers,
                    accuracy=None,
                    epsg_code=None,
                    reset_markers=False,
                    min_z=-50
                    ):
    """
    Add markers to an agisoft sfm project that represent ground control points with
    coordinates on pixel level but also with absolute coordinates.
    Args:
        chunk:
        markers:
        accuracy:
        epsg_code:
        reset_markers:
        min_z:

    Returns:

    """
    # remove all existing markers
    if reset_markers:
        chunk.remove(chunk.markers)
        chunk.markers.clear()

    # set crs of the markers
    if epsg_code is not None:
        chunk.marker_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa

    # multiply the number of gcps by the number of cameras to get the total number of possible markers
    ttl = markers.shape[0] * len(chunk.cameras)
    pbar = tqdm(total=ttl)

    # init marker variable
    marker = None

    # count number of markers
    num_markers = 0

    for _, row in markers.iterrows():

        # ignore markers with obviously wrong z values
        if row['z_abs'] < min_z:
            if debug_print_mode:
                print(f"Skip marker {row['GCP']} due to invalid z value")
            else:
                pbar.set_postfix_str(f"Skip marker {row['GCP']} due to invalid z value")
            continue

        # create 3d point
        point_3d = Metashape.Vector([row['x_rel'], row['y_rel'], row['z_rel']])  # noqa

        # transform the point to local coordinates
        point_local = chunk.transform.matrix.inv().mulp(point_3d)

        # reset the values
        marker = None
        skip_camera = False

        last_x = None
        last_y = None

        current_flight_path = None
        current_cam_view = None

        # iterate over the cameras
        for camera in chunk.cameras:

            # update progress bar
            pbar.update(1)

            # get the camera label
            camera_label = camera.label
            cam_flight_path = camera_label[2:6]
            cam_view = camera_label[6:8]

            # If flight path or direction changes, reset skipping logic
            if cam_flight_path != current_flight_path or cam_view != current_cam_view:
                skip_camera = False
                current_flight_path = cam_flight_path
                current_cam_view = cam_view
                last_x = None
                last_y = None

            if skip_camera:
                if debug_print_mode:
                    print(f"Skipping camera {camera_label}")
                else:
                    pbar.set_postfix_str(f"Skipping camera {camera_label}")
                continue

            # check if camera is aligned
            if not camera.transform:
                if debug_print_mode:
                    print(f"No transform for camera {camera.label}")
                else:
                    pbar.set_postfix_str("No transform for camera", camera.label)
                continue

            # project the point to the camera
            projection = camera.project(point_local)

            # skip invalid projections
            if projection is None:
                if debug_print_mode:
                    print(f"Projection for {camera.label} is invalid")
                else:
                    pbar.set_postfix_str(f"Projection for {camera.label} is invalid")
                continue

            # get the x and y coordinates
            x, y = projection

            # skip too small values (x)
            if x < 0:

                if last_x is None:
                    pass
                elif x < last_x:
                    # we are going away from the camera
                    skip_camera = True

                last_x = x
                last_y = y

                if debug_print_mode:
                    print(f"Projection for {camera.label} is negative ({x}, {y})")
                else:
                    pbar.set_postfix_str(f"Projection for {camera.label} is negative ({x}, {y})")
                continue

            # skip too small values (y)
            if y < 0:

                if last_y is None:
                    pass
                elif y < last_y:
                    # we are going away from the camera
                    skip_camera = True

                last_x = x
                last_y = y

                if debug_print_mode:
                    print(f"Projection for {camera.label} is negative ({x}, {y})")
                else:
                    pbar.set_postfix_str(f"Projection for {camera.label} is negative ({x}, {y})")
                continue

            # skip too big image (x)
            if x >= camera.image().width:

                if last_x is None:
                    pass
                elif x > last_x:
                    # we are going away from the camera
                    skip_camera = True

                last_x = x
                last_y = y

                if debug_print_mode:
                    print(f"Projection for {camera.label} is outside the image ({x}, {y})")
                else:
                    pbar.set_postfix_str(f"Projection for {camera.label} is outside the image ({x}, {y})")
                continue

            # skip too big image (y)
            if y >= camera.image().height:

                if last_y is None:
                    pass
                elif y > last_y:
                    # we are going away from the camera
                    skip_camera = True

                last_x = x
                last_y = y

                if debug_print_mode:
                    print(f"Projection for {camera.label} is outside the image ({x}, {y})")
                else:
                    pbar.set_postfix_str(f"Projection for {camera.label} is outside the image ({x}, {y})")
                continue

            # check if the x,y would be masked
            mask = camera.mask.image()
            if mask is not None:
                if mask[int(x), int(y)][0] == 0:
                    if debug_print_mode:
                        print(f"Projection for {camera.label} is masked ({x}, {y})")
                    else:
                        pbar.set_postfix_str(f"Projection for {camera.label} is masked ({x}, {y})")
                    continue

            # marker must be created only once
            if marker is None:
                num_markers += 1

                marker = chunk.addMarker()
                marker.label = row['GCP']

                # set the reference location for the marker
                marker.reference.location = Metashape.Vector(
                    [row['x_abs'], row['y_abs'], row['z_abs']])  # noqa

                # set the marker accuracy if available
                if accuracy is not None:
                    marker.reference.accuracy = Metashape.Vector([accuracy[0], accuracy[1], accuracy[2]])  # noqa

                if debug_print_mode:
                    print(f"Created marker {num_markers} at "
                          f"({row['x_abs']}, {row['y_abs']}, {row['z_abs']})")
                else:
                    pbar.set_postfix_str(f"Created marker {num_markers} at "
                          f"({row['x_abs']}, {row['y_abs']}, {row['z_abs']})")

            if debug_print_mode:
                print(f"Add marker {num_markers} to {camera.label} at {x}, {y}")
            else:
                pbar.set_postfix_str(f"Add marker {num_markers} to {camera.label} at {x}, {y}")

            # set relative projection for the marker
            m_proj = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)  # noqa
            marker.projections[camera] = m_proj  # noqa

            # reset the skip camera flag
            skip_camera = False
            last_x = None
            last_y = None

    pbar.close()

    # if the marker is still None something went wrong
    if marker is None:
        raise ValueError("No valid projections found for the marker")
    else:
        print(f"Added {num_markers} markers")

    return
