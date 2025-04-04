import Metashape
import numpy as np

import src.sfm_agi2.snippets.create_adapted_mask as cam

def union_masks(chunk, mask_folder):

    # iterate all cameras
    for camera in chunk.cameras:

        # skip cameras that are not enabled
        if camera.enabled is False:
            continue

        # check if camera has a mask
        if camera.mask is None:
            continue

        # get the mask
        mask = camera.mask.image()

        # get the dimensions of the mask
        m_width = mask.width
        m_height = mask.height

        # convert to np array
        mask_bytes = mask.tostring()
        existing_mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((m_height, m_width))

        # create an adapted mask
        adapted_mask = cam.create_adapted_mask(existing_mask, camera.label)

        # convert adapted mask to Metashape image
        adapted_mask_m = Metashape.Image(adapted_mask,
                                         adapted_mask.shape[1],
                                         adapted_mask.shape[0],
                                         channels=' ',
                                         datatype='U8')

        # create a mask object
        mask_obj = Metashape.Mask()
        mask_obj.setImage(adapted_mask_m)

        # set the mask to the camera
        camera.mask = mask_obj

        return