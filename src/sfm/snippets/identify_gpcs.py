import copy

import torch

import src.base.find_tie_points as ftp

def identify_gcps(images, transforms, overlap_dict=None):

    # init tie-point matcher
    tp_finder = ftp.TiePointDetector(matching_method="lightglue")

    # match every image with every image
    for i in range(len(images)):

        # find tie-points between the images
        tps, conf = tp_finder.find_tie_points(image1, image2)


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