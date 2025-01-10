import math
import Metashape


def filter_gcps(chunk,
                max_error_px=1, max_error_m=25,
                verbose=False):

    # infinite loop required; we break out when we are done
    while True:

        # save errors of the markers
        marker_errors_px = {}
        marker_errors_m = {}

        # check if there are any markers left
        if len(chunk.markers) == 0:
            raise ValueError("No markers left in the chunk")

        # iterate all markers
        for marker in chunk.markers:

            if verbose:
                print(f"Check marker '{marker.label}'")

            # check if marker is a GCP
            if 'gcp' not in marker.label:
                if verbose:
                    print(f" Skip marker '{marker.label}' as it is not a GCP")
                continue

            # check if marker has a position
            if marker.position is None:
                print(f" Skip marker '{marker.label}' as it has no position")
                continue

            # get position of marker
            position = marker.position

            # variables to calc marker error
            nr_projections = 0
            proj_sq_sum_px = 0
            proj_sq_sum_m = 0

            # iterate over all cameras
            for camera in marker.projections.keys():

                # skipping not aligned cameras
                if not camera.transform:
                    print(f"  Skip camera {camera.label} as it is not aligned")
                    continue

                # get real and estimate position (relative)
                proj = marker.projections[camera].coord  # noqa
                reproj = camera.project(position)

                # remove z coordinate from proj
                proj = Metashape.Vector([proj.x, proj.y])  # noqa

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
                print(f"  Skip marker '{marker.label}' as it has no projections")
                continue

            if verbose:
                print(" Error in px", marker_error_px)
                print(" Error in m", marker_error_m)

            # save errors to dict
            marker_errors_px[marker.label] = marker_error_px
            marker_errors_m[marker.label] = marker_error_m

        # get the marker with the highest error in px
        max_error_px_marker_name = max(marker_errors_px, key=marker_errors_px.get)
        max_error_px_value = marker_errors_px[max_error_px_marker_name]

        if max_error_px_value > max_error_px:
            # get the marker with the corresponding name
            marker = [m for m in chunk.markers if m.label in max_error_px_marker_name]

            # remove marker from chunk
            chunk.remove(marker)
            print(f"Remove marker '{max_error_px_marker_name}' with error {max_error_px_value} px")

            # update alignment and transform
            chunk.optimizeCameras()
            chunk.updateTransform()

            continue

        # get the marker with the highest error in m
        max_error_m_marker_name = max(marker_errors_m, key=marker_errors_m.get)
        max_error_m_value = marker_errors_m[max_error_m_marker_name]

        if max_error_m_value > max_error_m:
            # get the marker with the corresponding name
            marker = [m for m in chunk.markers if m.label in max_error_m_marker_name]

            # remove marker from chunk
            chunk.remove(marker)
            print(f"Remove marker '{max_error_m_marker_name}' with error {max_error_m_value} m")

            # update alignment and transform
            chunk.optimizeCameras()
            chunk.updateTransform()

            continue

        # if we reach this point, we are done
        break

    nr_markers = len(chunk.markers)
    print(f"{nr_markers} markers survived the purge")


if __name__ == "__main__":

    project_name = "test6"
    PROJECT_FLD = f"/data/ATM/data_1/sfm/agi_projects/{project_name}"
    import os
    project_path = os.path.join(PROJECT_FLD, project_name + ".psx")

    doc = Metashape.Document()
    doc.open(project_path)
    chunk = doc.chunk
    filter_gcps(chunk)