import numpy as np
import Metashape


def add_tp_markers(chunk, tp_dict, conf_dict,
                   reset_markers=False):

    # remove all existing markers
    if reset_markers:
        chunk.remove(chunk.markers)
        chunk.markers.clear()

    marker_id = 1

    # iterate over the tie-points
    for key in tp_dict:

        # get the camera names from the key
        cam1_name = key[0]
        cam2_name = key[1]

        # get the tie points
        tps = tp_dict[key]
        conf = conf_dict[key]

        print(tps.shape, conf.shape)

        # get the 10 tps with the highest confidence
        top_indices = np.argsort(conf)[-10:][::-1]
        top_tps = tps[top_indices]

        # create the markers
        for tp_row in top_tps:

            # create the marker
            marker = chunk.addMarker()
            marker.label = f"tp_marker_{marker_id}"

            # set local position for camera
            for camera in chunk.cameras:

                if camera.label == cam1_name:
                    x = tp_row[0]
                    y = tp_row[1]
                    m_proj = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)  # noqa
                elif camera.label == cam2_name:
                    x = tp_row[2]
                    y = tp_row[3]
                    m_proj = Metashape.Marker.Projection(Metashape.Vector([x, y]), True)  # noqa
                else:
                    continue

                marker.projections[camera] = m_proj  # noqa

            # add the marker to the chunk
            chunk.markers.append(marker)

            # increase the marker id
            marker_id += 1
