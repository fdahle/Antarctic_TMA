import copy
import torch

import base.print_v as p
import base.resize_image as ri

from external.SuperGlue.matching import Matching


def calc_image_complexity(image, highscore=2500, catch=True, verbose=False, pbar=None):
    """
    calc_image_complexity(image, highscore, catch, verbose, pbar):
    Calculate the complexity of an image as a number between 0 and 1, whereas 0 is a super simple image
    (completely white or black) and 1 means a highly complex image with many distinct structures.
    The complexity is based on the number of tie-points we can find with superglue
    Args:
        image (np-array): The image for that we want to calculate the highscore
        highscore (int): The number of tie-point when the score is one
        catch (Boolean): If true, we catch every error that is happening and return instead None
        verbose (Boolean): If true, we print information that is happening during execution of the
            function
        pbar (Tqdm-progressbar): If this is not None, the printing will be added to a tqdm-progress bar

    Returns:

    """

    p.print_v("Start: calc_image_complexity", verbose=verbose, pbar=pbar)

    try:
        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load & resize image
        sg_img = copy.deepcopy(image)
        sg_img = ri.resize_image(sg_img, (1000, 1000))
        sg_img = torch.from_numpy(sg_img)[None][None] / 255.
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

        data = matching.superpoint({'image': sg_img})
        data = {k + '0': data[k] for k in keys}

        kpts = data['keypoints0'][0].cpu().numpy()

        score = round(kpts.shape[0] / highscore, 2)
        score = min(score, 1)

        p.print_v("Finished: calc_image_complexity", verbose=verbose, pbar=pbar)
    except (Exception, ) as e:
        if catch:
            p.print_v("Failed: calc_image_complexity", verbose=verbose, pbar=pbar)
            return None
        else:
            raise e

    return score


if __name__ == "__main__":

    img_ids = ["CA135332V0305", "CA154932V0077", "CA171832V0082"]

    for img_id in img_ids:

        import base.load_image_from_file as liff
        img = liff.load_image_from_file(img_id)

        import base.remove_borders as rb
        img = rb.remove_borders(img, image_id=img_id)

        complexity = calc_image_complexity(img)

        print(complexity)

        import display.display_images as di
        di.display_images([img], title=complexity)
