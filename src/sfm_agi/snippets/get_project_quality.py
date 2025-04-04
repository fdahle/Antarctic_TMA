import math

import Metashape

def get_project_quality(chunk):

    check_cameras = True
    if check_cameras:

        aligned_images = 0

        for camera in chunk.cameras:

            if camera.transform is None:
                print(f"Camera {camera.label} is not aligned")
                continue
            aligned_images += 1
    else:
        aligned_images = None

    check_markers = True
    if check_markers:
        marker_errors_px = {}
        marker_errors_m = {}

        for marker in chunk.markers:

            if marker.enabled is False:
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
            marker_error_px = math.sqrt(proj_sq_sum_px / nr_projections)
            marker_error_m = math.sqrt(proj_sq_sum_m / nr_projections)

            # save errors to dict
            marker_errors_px[marker.label] = marker_error_px
            marker_errors_m[marker.label] = marker_error_m

        # get average errors
        avg_marker_error_px = sum(marker_errors_px.values()) / len(marker_errors_px)
        avg_marker_error_m = sum(marker_errors_m.values()) / len(marker_errors_m)
        nr_markers = len(marker_errors_px)



    else:
        nr_markers = None
        avg_marker_error_px = None
        avg_marker_error_m = None

    return_dict = {
        "aligned_images": aligned_images,
        "nr_markers": nr_markers,
        "marker_errors_px": avg_marker_error_px,
        "marker_errors_m": avg_marker_error_m,
    }

    return return_dict

if __name__ == "__main__":

    glacier = "pequod_glacier"
    project_psx_path = f"/data/ATM/data_1/sfm/agi_projects/{glacier}/{glacier}.psx"
    doc = Metashape.Document(read_only=False)  # noqa
    doc.open(project_psx_path, ignore_lock=True)

    chunk = doc.chunks[0]

    rd = get_project_quality(chunk)

    print(rd)