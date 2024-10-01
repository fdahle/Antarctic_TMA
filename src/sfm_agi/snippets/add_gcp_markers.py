import Metashape
from tqdm import tqdm


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
            continue

        # create 3d point
        point_3d = Metashape.Vector([row['x_rel'], row['y_rel'], row['z_rel']])  # noqa

        # transform the point to local coordinates
        point_local = chunk.transform.matrix.inv().mulp(point_3d)

        # reset the marker
        marker = None

        # iterate over the cameras
        for camera in chunk.cameras:

            # update progress bar
            pbar.update(1)

            # check if camera is aligned
            if not camera.transform:
                continue

            # project the point to the camera
            projection = camera.project(point_local)

            # skip invalid projections
            if projection is None:
                pbar.set_postfix_str(f"Projection for {camera.label} is invalid")
                continue

            # get the x and y coordinates
            x, y = projection

            # check if the point is within the image
            if ((0 <= x <= camera.image().width) and
                    (0 <= y <= camera.image().height)):

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

                # set relative projection for the marker
                m_proj = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)  # noqa
                marker.projections[camera] = m_proj  # noqa
            else:
                pbar.set_postfix_str(f"Projection for {camera.label} is outside the image")

    pbar.close()

    # if the marker is still None something went wrong
    if marker is None:
        raise ValueError("No valid projections found for the marker")
    else:
        print(f"Added {num_markers} markers")

    return
