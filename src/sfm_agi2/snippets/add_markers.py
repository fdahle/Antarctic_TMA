"""Adds Ground Control Point (GCP) markers to a Metashape chunk"""

import Metashape
import pandas as pd

from tqdm import tqdm

def add_markers(chunk: Metashape.Chunk,
                markers: pd.DataFrame,
                epsg_code: int,
                accuracy_dict: dict,
                reset_markers: bool = False,
                min_z: int =- 50,
                direct_projection: bool = False):
    """
    This function projects relative GCP coordinates into camera views,
    checks their validity, and creates markers with
    2D projections on the corresponding images.

    Args:
        chunk (Metashape.Chunk):
            The Metashape chunk to which the markers will be added.
        markers (pd.DataFrame):
            A DataFrame containing the GCP data with the following required columns:
            ['GCP', 'x_rel', 'y_rel', 'z_rel', 'x_abs', 'y_abs', 'z_abs'].
        epsg_code (int):
            The EPSG code for the marker coordinate system (e.g., 4326 for WGS 84).
        reset_markers (bool, optional):
            If True, removes all existing markers before adding new ones.
                Defaults to False.
        min_z (float, optional):
            Minimum threshold for absolute z value. Markers below this threshold are skipped. Defaults to -50.

    Raises:
        ValueError: If no valid projections are found for any marker.

    Returns:
        None
    """

    # Remove all existing markers if requested.
    if reset_markers:
        chunk.remove(chunk.markers)
        chunk.markers.clear()

    # Set the marker coordinate system
    chunk.marker_crs = Metashape.CoordinateSystem(f"EPSG::{epsg_code}")  # noqa

    # Counter for successfully created markers.
    num_markers = 0

    if direct_projection:

        pbar = tqdm(markers.itertuples(), total=markers.shape[0], desc="Directly adding GCP markers")
        camera_map = {camera.label: camera for camera in chunk.cameras}
        marker_map = {}

        for row in pbar:
            if row.z_abs < min_z:
                pbar.set_postfix_str(f"Skipping marker {row.GCP} due to invalid z")
                continue

            camera = camera_map.get(row.filename)
            if not camera:
                pbar.set_postfix_str(f"Camera {row.filename} not found")
                continue

            if not camera.transform:
                pbar.set_postfix_str(f"No transform for camera {row.filename}")
                continue

            if row.GCP not in marker_map:
                marker = chunk.addMarker()
                marker.label = row.GCP
                marker.reference.location = Metashape.Vector([row.x_abs, row.y_abs, row.z_abs])
                marker.reference.accuracy = Metashape.Vector([
                    row.accuracy_x if pd.notna(row.accuracy_x) else accuracy_dict['x'],
                    row.accuracy_y if pd.notna(row.accuracy_y) else accuracy_dict['y'],
                    row.accuracy_z if pd.notna(row.accuracy_z) else accuracy_dict['z']
                ])
                marker_map[row.GCP] = marker
                print(marker.label, marker.reference.location, marker.reference.accuracy)
                num_markers += 1

            marker = marker_map[row.GCP]
            m_proj = Metashape.Marker.Projection(Metashape.Vector([row.img_x, row.img_y]), True)
            marker.projections[camera] = m_proj

            pbar.set_postfix_str(f"Marker {row.GCP} added to {row.filename}")

    else:

        # Total iterations is the number of markers multiplied by the number of cameras.
        total_iterations = markers.shape[0] * len(chunk.cameras)
        pbar = tqdm(total=total_iterations, desc="Adding GCP markers")

        # init marker variable
        marker = None

        # extract x/y accuracy once
        x_acc = accuracy_dict["x"]
        y_acc = accuracy_dict["y"]
        z_acc = accuracy_dict["z"]

        # Process each marker row in the provided DataFrame.
        for _, row in markers.iterrows():
            # Skip markers with an absolute z value below the threshold.
            if row['z_abs'] < min_z:
                pbar.set_postfix_str(f"Skip marker {row['GCP']} due to invalid z value")
                pbar.update(len(chunk.cameras))
                continue

            # Create the 3D point from relative coordinates.
            point_3d = Metashape.Vector([row['x_rel'], row['y_rel'], row['z_rel']])  # noqa
            pbar.set_postfix_str(f"Point created at ({row['x_rel']}, "
                                 f"{row['y_rel']}, {row['z_rel']})")

            # Transform the point to the local coordinate system.
            point_local = chunk.transform.matrix.inv().mulp(point_3d)

            # reset the values
            marker = None
            skip_camera = False
            last_x, last_y = None, None
            current_flight_path, current_cam_view = None, None

            # Iterate over each camera in the chunk.
            for camera in chunk.cameras:
                camera_label = camera.label
                cam_flight_path = camera_label[2:6]
                cam_view = camera_label[6:8]

                # Reset the skip flag when the flight path or view changes.
                if cam_flight_path != current_flight_path or cam_view != current_cam_view:
                    skip_camera = False
                    current_flight_path = cam_flight_path
                    current_cam_view = cam_view
                    last_x, last_y = None, None

                if skip_camera:
                    pbar.set_postfix_str(f"Skipping camera {camera_label}")
                    pbar.update(1)
                    continue

                # Check for a valid camera transform.
                if not camera.transform:
                    pbar.set_postfix_str("No transform for camera {}".format(camera.label))
                    pbar.update(1)
                    continue

                # Project the local point into the camera image.
                projection = camera.project(point_local)
                if projection is None:
                    pbar.set_postfix_str(f"Projection for {camera.label} is invalid")
                    pbar.update(1)
                    continue

                x, y = projection

                # Check for negative coordinates or coordinates that suggest
                # the camera is moving away.
                if x < 0:
                    if last_x is not None and x < last_x:
                        skip_camera = True
                    last_x, last_y = x, y
                    pbar.set_postfix_str(f"Projection for {camera.label} is "
                                         f"negative ({x}, {y})")
                    pbar.update(1)
                    continue

                if y < 0:
                    if last_y is not None and y < last_y:
                        skip_camera = True
                    last_x, last_y = x, y
                    pbar.set_postfix_str(f"Projection for {camera.label} is "
                                         f"negative ({x}, {y})")
                    pbar.update(1)
                    continue

                # Check if the projection falls outside the image boundaries.
                img = camera.image()
                if x >= img.width:
                    if last_x is not None and x > last_x:
                        skip_camera = True
                    last_x, last_y = x, y
                    pbar.set_postfix_str(f"Projection for {camera.label} is outside "
                                         f"the image ({x}, {y})")
                    pbar.update(1)
                    continue

                if y >= img.height:
                    if last_y is not None and y > last_y:
                        skip_camera = True
                    last_x, last_y = x, y
                    pbar.set_postfix_str(f"Projection for {camera.label} is outside "
                                         f"the image ({x}, {y})")
                    pbar.update(1)
                    continue

                # If a mask exists for the camera, skip projections that fall
                # in the masked area.
                mask = camera.mask.image()
                if mask is not None:
                    if mask[int(x), int(y)][0] == 0:  # noqa
                        pbar.set_postfix_str(f"Projection for {camera.label} is "
                                             f"masked ({x}, {y})")
                        pbar.update(1)
                        continue

                # Create the marker (only once) if a valid projection is found.
                if marker is None:
                    num_markers += 1
                    marker = chunk.addMarker()
                    marker.label = row['GCP']
                    marker.reference.location = Metashape.Vector(
                        [row['x_abs'], row['y_abs'], row['z_abs']])  # noqa
                    pbar.set_postfix_str(f"Created marker {num_markers} at "
                                         f"({row['x_abs']}, {row['y_abs']}, {row['z_abs']})")

                    if z_acc == "auto":
                        z_acc = row.get("accuracy_z", 5.0)  # fallback if missing

                    marker.reference.accuracy = Metashape.Vector([x_acc, y_acc, z_acc])

                # Add the projection for the current camera to the marker.
                pbar.set_postfix_str(f"Add marker {num_markers} to "
                                     f"{camera.label} at {x}, {y}")
                m_proj = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)  # noqa
                marker.projections[camera] = m_proj  # noqa

                # Reset tracking variables for the next camera.
                skip_camera = False
                last_x, last_y = None, None
                pbar.update(1)

        pbar.close()

        if marker is None:
            raise ValueError("No valid projections found for any marker.")
        else:
            print(f"Added {num_markers} markers.")
