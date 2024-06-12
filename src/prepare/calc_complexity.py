"""Calculates the complexity of an image"""

# Library imports
import copy
import numpy as np
import torch
from typing import Optional

# External imports
from external.SuperGlue.matching import Matching

# Local imports
import src.base.resize_image as ri

# Constants
HIGHSCORE = 2500


def calc_complexity(image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Calculates the complexity of an image using the SuperGlue matching algorithm.
    Complexity is defined by the number of key-points detected in the image, normalized by a highscore value.
    An optional mask can be provided to specify areas of the image to include in the calculation.

    Args:
        image (np.ndarray): The image to analyze, as a numpy array.
        mask (Optional[np.ndarray]): An optional mask defining areas to include. Masked areas (0) are ignored,
                                      while unmasked areas (1) are considered. Must be the same shape as `image`.
    Returns:
        float: The complexity score of the image, a value between 0 and 1.
    Raises:
        ValueError: If `image` and `mask` shapes do not match.
    """

    # check mask validity
    if mask is not None:
        if image.shape != mask.shape:
            raise ValueError("Image and mask must have the same shape.")

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # deep copy image and mask
    image = copy.deepcopy(image)
    if mask is not None:
        mask = copy.deepcopy(mask)

    # resize image and mask
    image = ri.resize_image(image, (1000, 1000))
    if mask is not None:
        mask = ri.resize_image(mask, (1000, 1000))

    # put to torch
    sg_img = torch.from_numpy(image)[None][None] / 255.
    sg_img = sg_img.to(device)

    # superglue settings
    nms_radius = 3
    keypoint_threshold = 0.005
    max_keypoints = -1  # -1 keep all
    weights = "outdoor"  # can be indoor or outdoor
    sinkhorn_iterations = 20
    match_threshold = 0.2

    # set config for superglue
    superglue_config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': weights,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

    # init the matcher
    matching = Matching(superglue_config).eval().to(device)

    # what do we want to detect
    keys = ['keypoints', 'scores', 'descriptors']

    # detect keypoints
    data = matching.superpoint({'image': sg_img})
    data = {k + '0': data[k] for k in keys}
    kpts = data['keypoints0'][0].cpu().numpy()

    # filter tie-points if mask is provided
    if mask is not None:
        kpts = kpts[np.where(mask[(kpts[:, 1]).astype(int), (kpts[:, 0]).astype(int)] == 1)]

    score = round(kpts.shape[0] / HIGHSCORE, 2)
    score = min(score, 1)

    return score
