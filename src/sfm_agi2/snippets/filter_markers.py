"""" Filters markers in a chunk """

import math
import Metashape

from src.sfm_agi2.SfMError import SfMError


def filter_markers(chunk: Metashape.Chunk,
                   marker_lbl: str = 'gcp',
                   min_markers: int = 3,
                   max_error_px: int = 1, max_error_m: int = 25) -> None:
    """
    The function iteratively evaluates each GCP marker in the given chunk by
    computing its average reprojection error in pixels and its metric error in meters.
    If a marker's error exceeds the given thresholds, the marker is disabled,
    the cameras are re-optimized, and the transform updated. The process continues
    until no marker exceeds the specified error thresholds or the number of enabled
    markers falls below or equals the minimum allowed.

    Args:
        chunk (Metashape.chunk): The Metashape chunk object containing markers, cameras, and transforms.
        min_markers (int, optional): Minimum number of enabled markers
            required to stop filtering. Defaults to 3.
        max_error_px (float, optional): Maximum allowed average reprojection
            error in pixels. Defaults to 1.0.
        max_error_m (float, optional): Maximum allowed average metric error
            in meters. Defaults to 25.0.

    Returns:
        None
    """

    # infinite loop to remove markers
    while True:

        # save errors of the markers
        marker_errors_px = {}
        marker_errors_m = {}

        # get number of enabled markers
        enabled_markers = [m for m in chunk.markers if m.enabled and marker_lbl in m.label]
        nr_markers = len(enabled_markers)

        print(f"Number of enabled markers: {nr_markers}")

        # stop the loop
        if nr_markers <= min_markers:
            print("Stop filtering as minimum number of markers ({}) is reached".format(
                min_markers))
            break

        # iterate all markers
        for marker in enabled_markers:

            if marker.position is None:
                continue

            marker_error_px, marker_error_m = _calc_marker_error(chunk, marker)

            # Skip invalid markers
            if marker_error_px is None or marker_error_m is None:
                print("[WARNING] Invalid marker with no projections")
                continue

            # save errors to dict
            marker_errors_px[marker.label] = marker_error_px
            marker_errors_m[marker.label] = marker_error_m

        if len(marker_errors_px) == 0:
            raise SfMError("No markers found in chunk with label '{}'".format(marker_lbl))

        # get the marker with the highest error in px
        max_error_px_marker_name = max(marker_errors_px, key=marker_errors_px.get)
        max_error_px_value = marker_errors_px[max_error_px_marker_name]

        if max_error_px_value > max_error_px:
            # get the marker with the corresponding name
            marker = [m for m in chunk.markers if m.label == max_error_px_marker_name]

            # disable marker
            for m in marker:
                m.enabled = False

            print(f"Remove marker '{max_error_px_marker_name}' with "
                  f"error {max_error_px_value} px")

            # update alignment and transform
            chunk.optimizeCameras()
            chunk.updateTransform()

            continue

        # get the marker with the highest error in m
        max_error_m_marker_name = max(marker_errors_m, key=marker_errors_m.get)
        max_error_m_value = marker_errors_m[max_error_m_marker_name]

        if max_error_m_value > max_error_m:
            # get the marker with the corresponding name
            marker = [m for m in chunk.markers if m.label ==
                      max_error_m_marker_name]

            # remove marker from chunk
            for m in marker:
                m.enabled = False

            print(f"Remove marker '{max_error_m_marker_name}' with "
                  f"error {max_error_m_value} m")

            # update alignment and transform
            chunk.optimizeCameras()
            chunk.updateTransform()

            continue

        # if we reach this point, we are done
        break

    # just raise an error if we have no markers left
    if len(chunk.markers) == 0:
        raise SfMError("No markers left in chunk after filtering")

    nr_markers = len(chunk.markers)
    print(f"{nr_markers} markers survived the purge")


def _calc_marker_error(chunk: Metashape.Chunk,
                       marker: Metashape.Marker):
    # get position of marker
    position = marker.position

    # variables to calc marker error
    nr_projections = 0
    proj_sq_sum_px = 0
    proj_sq_sum_m = 0

    for camera in marker.projections.keys():
        # skipping not aligned cameras
        if not camera.transform:
            continue

        # get real and estimate position (relative)
        proj = marker.projections[camera].coord  # noqa
        reproj = camera.project(position)

        # remove z coordinate from proj
        proj = Metashape.Vector([proj.x, proj.y])  # noqa

        if proj is None or reproj is None:
            continue

        # calculate px error
        error_norm_px = (reproj - proj).norm()

        # calculate squared error (relative)
        proj_sq_sum_px += error_norm_px ** 2


        # get real and estimate position (absolute)
        est = chunk.crs.project(
            chunk.transform.matrix.mulp(marker.position))
        ref = marker.reference.location

        # calculate m error
        error_norm_m = (est - ref).norm()  # noqa

        # calculate squared error (absolute)
        proj_sq_sum_m += error_norm_m ** 2

        # update number of projections
        nr_projections += 1

    # get average errors
    if nr_projections > 0:
        marker_error_px = math.sqrt(proj_sq_sum_px / nr_projections)
        marker_error_m = math.sqrt(proj_sq_sum_m / nr_projections)
    else:
        marker_error_px = None
        marker_error_m = None

    return marker_error_px, marker_error_m