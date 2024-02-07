import copy
import numpy as np
import torch

import src.base.find_tie_points as ftp

def identify_gcps(images, transforms):

    # init tie-point matcher
    tp_finder = ftp.TiePointDetector(matching_method="lightglue")

    all_gcps = np.empty([0, 4]
    all_conf = np.empty([0, 1])

    # iterate all images
    for image in images):

        # find gcps in the images
        tps, conf = _find_gcps(image)


def _find_gcps(image):
    sg_image = copy.deepcopy(image)
    sg_image = torch.from_numpy(sg_image)[None][None] / 255.

    # init device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load extractor and find points
    extractor = SuperPoint(max_num_keypoints=None).eval().to(device)
    features = extractor.extract(sg_image.to(device))

    key_points

if __name__ == "__main__":

    images = ["ID1", "ID2", "ID3", "ID4", "ID5"]
    identify_gcps(images, "")