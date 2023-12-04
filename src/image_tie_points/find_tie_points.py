import copy
import cv2 as cv2
import numpy as np
import torch
import math
import warnings

from skimage import transform as tf

from external.SuperGlue.matching import Matching
from external.lightglue import LightGlue, SuperPoint
from external.lightglue.utils import rbd

import base.print_v as p

import display.display_images as di
import display.display_tiepoints as dt

debug_show_input_images = False  # the resized images
debug_show_input_masks = False  # the resized masks
debug_show_initial_tiepoints_small = False  # the tiepoints found with resized images
                                            # (but not already back to full size)
debug_show_initial_tiepoints = False  # the tiepoints found with resized images
debug_show_additional_subset = False  # show the subset with additional tiepoints
debug_show_additional_tiles = False  # show the images withe additional tiles
debug_show_additional_tiepoints = False  # the additional tiepoints
debug_show_transformation_points = False  # the tiepoints used for the transformation matrix
debug_show_extra_subset = False
debug_show_extra_tiles = False
debug_show_extra_tiepoints = False  # the extra tiepoints
debug_show_final_tiepoints = False  # the tie-points after all matching is done

debug_print_trans_mat = False


def find_tie_points(input_img1, input_img2, mask_1=None, mask_2=None,
                    max_width=2000, max_height=1500,
                    extra_mask_padding=0,
                    additional_matching=True,
                    extra_matching=False,
                    min_threshold=None,
                    filter_outliers=True,
                    keep_resized_points=False,
                    keep_additional_points=True,
                    debug_clean_tie_points=True,
                    clean_threshold=10,
                    matching_method="LightGlue",
                    catch=True, verbose=False, pbar=None):
    """
    find_tie_points:
    This function extracts tie-points between two images making use of SuperGlue.
    First an initial matching on resized imagery is done. Afterwards, subsets are extracted around
    the location of the tie-points found during resizing. A second matching is then done using
    these subsets in original resolution. The size of these subsets is calculated automatically,
    depending on the available memory in the GPU. Additional matching: After tie-points are
    found, the transformation matrix between the two images is calculated. This is used to
    check where a subset of one image is located in the other image. Between these subsets,
    additional matching is done and added to the final results.
    Args:
        input_img1 (np-array): One image used for tie-point matching.
        input_img2 (np-array): Second image used for tie-point matching.
        mask_1 (np-array): Mask for input_img1, binary np array
            (0 means points at that location are filtered).
        mask_2 (np-array): Mask for input_img2, binary np array
            (0 means points at that location are filtered).
        max_width (int): width for the resized image and the initial subset
        max_height (int): height for the resized image and the initial subset
        extra_mask_padding (int): The number of pixels that are subtracted at the edge of
            the mask (When using image enhancements, there are many false tie-points at the edges)
        additional_matching (Boolean): If true, an additional matching is done
            (see description at the beginning)
        extra_matching (Boolean): If true, an extra matching is done
        min_threshold (float): The minimum quality required for the tie-points
            (Value between 0 and 1). If none, all tie-points are saved regardless of the quality.
        filter_outliers (Boolean): If true, RANSAC is applied to filter outliers of
            the tie-points.
        keep_resized_points (Boolean): If true, the tie-points find during matching
            of the resized images are kept as well.
        keep_additional_points (Boolean): If true, the tie points found during the
            additional matching are kept
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
    all_tiepoints (np-array): Array with all the tie-points (X1, Y1, X2, Y2)
    all_confidences (np-array): List of confidence values of the tie_points
    """

    p.print_v("Start: find_tie_points", verbose, pbar=pbar)

    # deep copy to not change the original images
    img_1 = copy.deepcopy(input_img1)
    img_2 = copy.deepcopy(input_img2)

    # images must be in grayscale
    if len(img_1.shape) == 3:
        if img_1.shape[0] == 3:
            img_1 = np.moveaxis(img_1, 0, -1)
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    if len(img_2.shape) == 3:
        if img_2.shape[0] == 3:
            img_2 = np.moveaxis(img_2, 0, -1)
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # check the inputs
    if mask_1 is not None:
        assert img_1.shape[0] == mask_1.shape[0] and img_1.shape[1] == mask_1.shape[1]
    if mask_2 is not None:
        assert img_2.shape[0] == mask_2.shape[0] and img_2.shape[1] == mask_2.shape[1]
    assert extra_mask_padding >= 0
    assert max_width >= 0
    assert max_height >= 0

    # init device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init the matching algorithms
    if matching_method == "LightGlue":

        extractor = SuperPoint(max_num_keypoints=None).eval().to(device)  # load the extractor
        matcher = LightGlue(features='superpoint', depth_confidence=-1, width_confidence=-1).eval().to(device)

    elif matching_method == "SuperGlue":

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

    else:
        raise ValueError(f"No matching method found with '{matching_method}'")

    try:

        # function to resize the images
        def resize_img(img, height_max, width_max):

            # check if we need to resize the images
            if img.shape[0] > height_max or img.shape[1] > width_max:

                # set resized flag
                bool_resized = True

                # check if we need to resize due to width or height
                if img.shape[0] >= img.shape[1]:
                    resize_factor = height_max / img.shape[0]
                else:
                    resize_factor = width_max / img.shape[1]

                # get the new image shape and resize
                new_image_shape = (int(img.shape[0] * resize_factor), int(img.shape[1] * resize_factor))
                img_resized = cv2.resize(img, (new_image_shape[1], new_image_shape[0]), interpolation=cv2.INTER_NEAREST)

            else:
                # set resized flag
                bool_resized = False

                # we don't need to change the image
                img_resized = img
                resize_factor = 1

            return bool_resized, img_resized, resize_factor

        # function to actual get tie points with LightGlue
        def apply_lightglue(input_img_1, input_img_2):

            # put stuff to cuda
            sg_img_1 = copy.deepcopy(input_img_1)
            sg_img_2 = copy.deepcopy(input_img_2)

            sg_img_1 = torch.from_numpy(sg_img_1)[None][None] / 255.
            sg_img_2 = torch.from_numpy(sg_img_2)[None][None] / 255.

            # extract features
            feats0 = extractor.extract(sg_img_1.to(device))
            feats1 = extractor.extract(sg_img_2.to(device))

            # get matches
            matches01 = matcher({'image0': feats0, 'image1': feats1})
            feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

            kpts0, kpts1, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

            pts0 = m_kpts0.cpu().numpy()
            pts1 = m_kpts1.cpu().numpy()
            conf = matches01['scores'].detach().cpu().numpy()

            return pts0, pts1, conf

        # function to actual get tie points with SuperGlue
        def apply_superglue(input_img_1, input_img_2):

            # put stuff to cuda
            sg_img_1 = copy.deepcopy(input_img_1)
            sg_img_2 = copy.deepcopy(input_img_2)

            sg_img_1 = torch.from_numpy(sg_img_1)[None][None] / 255.
            sg_img_2 = torch.from_numpy(sg_img_2)[None][None] / 255.

            sg_img_1 = sg_img_1.to(device)
            sg_img_2 = sg_img_2.to(device)

            last_data = matching.superpoint({'image': sg_img_1})
            last_data = {k + '0': last_data[k] for k in keys}
            last_data["image0"] = sg_img_1

            pred = matching({**last_data, 'image1': sg_img_2})
            kpts0 = last_data['keypoints0'][0].cpu().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            matches_superglue = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].detach().cpu().numpy()

            # keep the matching key points
            valid = matches_superglue > -1
            mkpts_0 = kpts0[valid]
            mkpts_1 = kpts1[matches_superglue[valid]]
            m_conf = confidence[valid]

            return mkpts_0, mkpts_1, m_conf

        # function to filter tie points with masks
        def filter_with_mask(f_mktps, f_conf, f_mask1, f_mask2):

            # if both filters are None, the filtering is easy-peasy
            if f_mask1 is None and f_mask2 is None:
                return f_mktps, f_conf, None

            num_original_tps = f_mktps.shape[0]

            # init array where we store the points we want to filter
            filter_values_1, filter_values_2 = [], []

            for row in f_mktps:
                if f_mask1 is None:
                    filter_values_1.append(1)
                else:

                    if extra_mask_padding == 0:
                        filter_values_1.append(f_mask1[int(row[1]), int(row[0])])
                    else:

                        _min_subset_x = row[0] - extra_mask_padding
                        _max_subset_x = row[0] + extra_mask_padding
                        _min_subset_y = row[1] - extra_mask_padding
                        _max_subset_y = row[1] + extra_mask_padding

                        if _min_subset_x < 0:
                            _min_subset_x = 0
                        if _min_subset_y < 0:
                            _min_subset_y = 0
                        if _max_subset_x > f_mask1.shape[1]:
                            _max_subset_x = f_mask1.shape[1]
                        if _max_subset_y > f_mask1.shape[0]:
                            _max_subset_y = f_mask1.shape[0]

                        _min_subset_x = int(_min_subset_x)
                        _min_subset_y = int(_min_subset_y)
                        _max_subset_x = int(_max_subset_x)
                        _max_subset_y = int(_max_subset_y)

                        mask_subset = f_mask1[_min_subset_y:_max_subset_y, _min_subset_x:_max_subset_x]

                        if 0 in mask_subset:
                            filter_values_1.append(0)
                        else:
                            filter_values_1.append(1)

                if f_mask2 is None:
                    filter_values_2.append(1)
                else:

                    if extra_mask_padding == 0:
                        filter_values_2.append(f_mask2[int(row[3]), int(row[2])])
                    else:

                        _min_subset_x = row[2] - extra_mask_padding
                        _max_subset_x = row[2] + extra_mask_padding
                        _min_subset_y = row[3] - extra_mask_padding
                        _max_subset_y = row[3] + extra_mask_padding

                        if _min_subset_x < 0:
                            _min_subset_x = 0
                        if _min_subset_y < 0:
                            _min_subset_y = 0
                        if _max_subset_x > f_mask2.shape[1]:
                            _max_subset_x = f_mask2.shape[1]
                        if _max_subset_y > f_mask2.shape[0]:
                            _max_subset_y = f_mask2.shape[0]

                        _min_subset_x = int(_min_subset_x)
                        _min_subset_y = int(_min_subset_y)
                        _max_subset_x = int(_max_subset_x)
                        _max_subset_y = int(_max_subset_y)

                        mask_subset = f_mask2[_min_subset_y:_max_subset_y, _min_subset_x:_max_subset_x]

                        if 0 in mask_subset:
                            filter_values_2.append(0)
                        else:
                            filter_values_2.append(1)

            # convert to np array
            filter_values_1 = np.asarray(filter_values_1)
            filter_values_2 = np.asarray(filter_values_2)

            _filter_indices = np.logical_or(filter_values_1 == 0, filter_values_2 == 0)
            _filter_indices = 1 - _filter_indices
            _filter_indices = _filter_indices.astype(bool)

            f_mkpts = f_mktps[_filter_indices, :]
            f_conf = f_conf[_filter_indices]

            p.print_v(f"{num_original_tps - f_mkpts.shape[0]} of {num_original_tps} tie-points are masked",
                      verbose=verbose, pbar=pbar)

            return f_mkpts, f_conf, _filter_indices

        # check and resize the images
        bool_resized_1, img_1_resized, resize_factor_1 = resize_img(img_1, max_height, max_width)
        bool_resized_2, img_2_resized, resize_factor_2 = resize_img(img_2, max_height, max_width)

        if debug_show_input_images:
            di.display_images([img_1_resized, img_2_resized])

        if debug_show_input_masks:
            _, mask_1_small, _ = resize_img(mask_1, img_1_resized.shape[0], img_1_resized.shape[1])
            _, mask_2_small, _ = resize_img(mask_2, img_2_resized.shape[0], img_2_resized.shape[1])
            di.display_images([mask_1_small, mask_2_small])

        # we try getting tie-points as long as we get results and not oom-error
        while True:
            try:

                # get the first batch of tie points for a smaller image
                if matching_method == "LightGlue":
                    mkpts1, mkpts2, mconf = apply_lightglue(img_1_resized, img_2_resized)
                elif matching_method == "SuperGlue":
                    mkpts1, mkpts2, mconf = apply_superglue(img_1_resized, img_2_resized)

                # that means we didn't raise an error but found a solution
                break

            except (Exception,) as e:
                if "out of memory" in str(e):

                    # free the gpu
                    torch.cuda.empty_cache()

                    # calculate new height and width
                    max_width = int(0.9 * max_width)
                    max_height = int(0.9 * max_height)

                    # resize images again
                    bool_resized_1, img_1_resized, resize_factor_1 = resize_img(img_1, max_height, max_width)
                    bool_resized_2, img_2_resized, resize_factor_2 = resize_img(img_2, max_height, max_width)

                    p.print_v(f"Out of memory, try again with new image size for "
                              f"img1 {img_1_resized.shape} and img2 {img_2_resized.shape}", verbose=verbose, pbar=pbar)
                else:
                    raise e

        p.print_v(f"initial matching done ({mkpts1.shape[0]} tie-points found)", verbose=verbose, pbar=pbar)

        # merge the tiepoints of left and right image
        mkpts = np.concatenate((mkpts1, mkpts2), axis=1)

        if debug_show_initial_tiepoints_small:
            # create temporary copies
            mkpts_small = copy.deepcopy(mkpts)
            mconf_small = copy.deepcopy(mconf)
            mkpts_small = mkpts_small.astype(int)
            _, mask_1_small, _ = resize_img(mask_1, img_1_resized.shape[0], img_1_resized.shape[1])
            _, mask_2_small, _ = resize_img(mask_2, img_2_resized.shape[0], img_2_resized.shape[1])

            mkpts_small, mconf_small, filter_indices_small = filter_with_mask(mkpts_small, mconf_small,
                                                                              mask_1_small, mask_2_small)

            dt.display_tiepoints([img_1_resized, img_2_resized], mkpts_small, mconf_small)

        # fit tie points to the big image again
        mkpts[:, 0] = mkpts[:, 0] * 1 / resize_factor_1
        mkpts[:, 1] = mkpts[:, 1] * 1 / resize_factor_1
        mkpts[:, 2] = mkpts[:, 2] * 1 / resize_factor_2
        mkpts[:, 3] = mkpts[:, 3] * 1 / resize_factor_2
        mkpts = mkpts.astype(int)

        # apply threshold if wished
        if min_threshold is not None:
            num_tp_before = mkpts.shape[0]
            mkpts = mkpts[mconf >= min_threshold]
            mconf = mconf[mconf >= min_threshold]
            num_tp_after = mkpts.shape[0]

            p.print_v(f"{num_tp_before - num_tp_after} tiepoints removed with a quality lower than {min_threshold}.",
                      verbose=verbose, pbar=pbar)

        # filter for outliers
        if filter_outliers and mkpts.shape[0] >= 4:
            _, mask = cv2.findHomography(mkpts[:, 0:2], mkpts[:, 2:4], cv2.RANSAC, 5.0)
            mask = mask.flatten()

            p.print_v(f"{np.count_nonzero(mask)} outliers removed.", verbose=verbose, pbar=pbar)

            # 1 means outlier
            mkpts = mkpts[mask == 0]
            mconf = mconf[mask == 0]

        # mask the points if necessary
        mkpts_all = copy.deepcopy(mkpts)
        mconf_all = copy.deepcopy(mconf)
        mkpts, mconf, filter_indices = filter_with_mask(mkpts, mconf, mask_1, mask_2)

        # no tie points found unfortunately
        if mkpts.shape[0] == 0:
            # return the empty shapes
            p.print_v("Warning: No tie-points found!", verbose=verbose, pbar=pbar)
            return mkpts, mconf

        if debug_show_initial_tiepoints:
            dt.display_tiepoints([input_img1, input_img2], mkpts_all, mconf_all,
                                 filter_indices=filter_indices,
                                 title=f"initial tiepoints {mkpts.shape[0]} "
                                       f"({mkpts_all.shape[0] - mkpts.shape[0]} masked)")

        # if images were resized we can do some more stuff to find good tie points
        if additional_matching is True and (bool_resized_1 is True or bool_resized_2 is True):

            p.print_v("Start additional matching:", verbose=verbose, pbar=pbar)

            # we need to iterate over the bigger image & check which image is bigger
            switch_base_other = False
            if img_1.shape[0] * img_1.shape[1] < img_2.shape[0] * img_2.shape[1]:
                switch_base_other = True

            if switch_base_other is False:
                base_img = img_1
                other_img = img_2
                mkpts_base = mkpts[:, 0:2]
                mkpts_other = mkpts[:, 2:4]
            else:
                base_img = img_2
                other_img = img_1
                mkpts_base = mkpts[:, 2:4]
                mkpts_other = mkpts[:, 0:2]

            # how often does the maximum possible image fit in our base img
            num_x = math.ceil(base_img.shape[1] / max_width)
            num_y = math.ceil(base_img.shape[0] / max_height)

            # how much do we have too much
            too_much_x = num_x * max_width - base_img.shape[1]
            too_much_y = num_y * max_height - base_img.shape[0]

            # how much we need to shift the image every step
            if num_x - 1 == 0:
                reduce_x = 0
            else:
                reduce_x = int(too_much_x / (num_x - 1))
            if num_y - 1 == 0:
                reduce_y = 0
            else:
                reduce_y = int(too_much_y / (num_y - 1))

            # here we save all tiepoints
            additional_tiepoints = []
            additional_confidences = []

            # here we save the tiles (to display them later)
            tiles = []
            other_tiles = []

            # init the progress bar
            max_counter = math.ceil(num_y) * math.ceil(num_x)
            add_counter = 0

            # iterate over the tiles
            for y_counter in range(0, math.ceil(num_y)):
                for x_counter in range(0, math.ceil(num_x)):

                    add_counter += 1
                    p.print_v(f"Additional matching: {add_counter}/{max_counter}", verbose=verbose, pbar=pbar)

                    # calculate the extent of the tile
                    min_x = x_counter * max_width - reduce_x * x_counter
                    max_x = (x_counter + 1) * max_width - reduce_x * x_counter
                    min_y = y_counter * max_height - reduce_y * y_counter
                    max_y = (y_counter + 1) * max_height - reduce_y * y_counter

                    # get the indices of the points & points itself that are in this tile
                    base_indices = np.where((mkpts_base[:, 0] >= min_x) & (mkpts_base[:, 0] <= max_x) &
                                            (mkpts_base[:, 1] >= min_y) & (mkpts_base[:, 1] <= max_y))
                    base_points = mkpts_base[base_indices]

                    # if we don't have points -> continue
                    if len(base_points) == 0:
                        p.print_v("No tie-points to match", verbose=verbose, pbar=pbar)
                        continue

                    # based on these indices get the points of the "other" image
                    other_points = mkpts_other[base_indices]

                    # init step for the percentile loop
                    step = 0

                    # get tie points based on percentiles, do it so long until we found a good image
                    while True:

                        # calculate the percentile of these selected points
                        other_percentile_sm = np.percentile(other_points, step, axis=0).astype(int)
                        other_percentile_la = np.percentile(other_points, 100 - step, axis=0).astype(int)

                        # calculate extent of the subset (min_x, max_y, min_y, max_y
                        min_subset_x = other_percentile_sm[0]  # - int(max_width / 2)
                        max_subset_x = other_percentile_la[0]  # + int(max_width / 2)
                        min_subset_y = other_percentile_sm[1]  # - int(max_height / 2)
                        max_subset_y = other_percentile_la[1]  # + int(max_height / 2)

                        # if extent is smaller than what we could -> enlarge
                        if max_subset_x - min_subset_x < max_width:
                            missing_width = max_width - (max_subset_x - min_subset_x)
                            min_subset_x = min_subset_x - int(missing_width / 2)
                            max_subset_x = max_subset_x + int(missing_width / 2)
                        if max_subset_y - min_subset_y < max_height:
                            missing_height = max_height - (max_subset_y - min_subset_y)
                            min_subset_y = min_subset_y - int(missing_height / 2)
                            max_subset_y = max_subset_y + int(missing_height / 2)

                        # make absolutely sure that we are not getting out of image bounds
                        # should not be necessary, but better safe than sorry
                        min_subset_x = max(0, min_subset_x)
                        min_subset_y = max(0, min_subset_y)
                        max_subset_x = min(other_img.shape[1], max_subset_x)
                        max_subset_y = min(other_img.shape[0], max_subset_y)

                        # the tie points we found are good, because the subset is small enough!
                        if max_subset_x - min_subset_x <= max_width and \
                                max_subset_y - min_subset_y <= max_height:
                            break

                        # with the current step we found too many tie-points
                        # -> decrease the range in which we look for tie points
                        step = step + 5

                    # get the subsets for superglue extraction
                    base_subset = base_img[min_y:max_y, min_x:max_x]
                    other_subset = other_img[min_subset_y:max_subset_y, min_subset_x:max_subset_x]

                    # save the tiles for later to show them
                    tiles.append([min_x, max_x, min_y, max_y])
                    other_tiles.append([min_subset_x, max_subset_x, min_subset_y, max_subset_y])

                    # init sub_resize factors so that they exist
                    sub_resize_factor_1, sub_resize_factor_2 = 1, 1

                    # get the tie points for these subsets
                    while True:
                        try:

                            if matching_method == "LightGlue":
                                mkpts1_sub, mkpts2_sub, mconf_sub = apply_lightglue(base_subset, other_subset)
                            elif matching_method == "SuperGlue":
                                mkpts1_sub, mkpts2_sub, mconf_sub = apply_superglue(base_subset, other_subset)

                            # that means we didn't raise an error but found a solution
                            break

                        except (Exception,) as e:
                            if "out of memory" in str(e):

                                # free the gpu
                                torch.cuda.empty_cache()

                                # calculate new height and width
                                new_base_width = int(0.9 * base_subset.shape[1])
                                new_base_height = int(0.9 * base_subset.shape[0])

                                new_other_width = int(0.9 * other_subset.shape[1])
                                new_other_height = int(0.9 * other_subset.shape[0])

                                # resize images again
                                _, base_subset, sub_resize_factor_1 = resize_img(base_subset, new_base_height,
                                                                                 new_base_width)
                                _, other_subset, sub_resize_factor_2 = resize_img(other_subset, new_other_height,
                                                                                  new_other_width)

                                p.print_v(f"Out of memory, new size ({base_subset.shape}, {other_subset.shape})",
                                          verbose=verbose, pbar=pbar)
                            else:
                                raise e

                    # merge the tie points
                    mkpts_sub = np.concatenate((mkpts1_sub, mkpts2_sub), axis=1)

                    if debug_show_additional_subset:
                        # create temp mktps to show in display_tiepoints
                        temp_mkpts = np.concatenate((mkpts_base[base_indices], mkpts_other[base_indices]), axis=1)
                        temp_mkpts = copy.deepcopy(temp_mkpts)
                        temp_conf = mconf[base_indices]

                        # adapt the tie points of the other image to account that we look at a subset
                        temp_mkpts[:, 0] = temp_mkpts[:, 0] - min_x
                        temp_mkpts[:, 1] = temp_mkpts[:, 1] - min_y
                        temp_mkpts[:, 2] = temp_mkpts[:, 2] - min_subset_x
                        temp_mkpts[:, 3] = temp_mkpts[:, 3] - min_subset_y

                        dt_type = 'new'  # can be 'original' or 'new'

                        if y_counter == 4 and x_counter == 3:

                            if dt_type == "original":
                                dt.display_tiepoints([base_subset, other_subset],
                                                     points=temp_mkpts, confidences=temp_conf,
                                                     title=f"original ({temp_mkpts.shape[0]}) & "
                                                           f"new tiepoints ({mkpts_sub.shape[0]}) for "
                                                           f"subset ({y_counter}/{x_counter})")
                            elif dt_type == "new":
                                dt.display_tiepoints([base_subset, other_subset],
                                                     points=mkpts_sub, confidences=mconf_sub,
                                                     title=f"original ({temp_mkpts.shape[0]}) & "
                                                           f"new tiepoints ({mkpts_sub.shape[0]}) for "
                                                           f"subset ({y_counter}/{x_counter})")

                    # adapt the tie points to account for resizing
                    mkpts_sub[:, 0] = mkpts_sub[:, 0] * 1 / sub_resize_factor_1
                    mkpts_sub[:, 1] = mkpts_sub[:, 1] * 1 / sub_resize_factor_1
                    mkpts_sub[:, 2] = mkpts_sub[:, 2] * 1 / sub_resize_factor_2
                    mkpts_sub[:, 3] = mkpts_sub[:, 3] * 1 / sub_resize_factor_2
                    mkpts = mkpts.astype(int)

                    # adapt the tie points of the other image to account that we look at a subset
                    mkpts_sub[:, 0] = mkpts_sub[:, 0] + min_x
                    mkpts_sub[:, 1] = mkpts_sub[:, 1] + min_y
                    mkpts_sub[:, 2] = mkpts_sub[:, 2] + min_subset_x
                    mkpts_sub[:, 3] = mkpts_sub[:, 3] + min_subset_y

                    # if we switched the images we need to switch the tie points
                    if switch_base_other:
                        temp = np.column_stack((mkpts_sub[:, 2], mkpts_sub[:, 3], mkpts_sub[:, 0], mkpts_sub[:, 1]))
                        mkpts_sub = temp

                    # add the tie points and confidence values to the list
                    additional_tiepoints.append(mkpts_sub)
                    for elem in mconf_sub:
                        additional_confidences.append(elem)

                    p.print_v(f"{mkpts_sub.shape[0]} tie-points matched", verbose=verbose, pbar=pbar)

            # reset progress bar
            p.print_v("Additional matching finished", verbose=verbose, pbar=pbar)

            if debug_show_additional_tiles:

                tiles_reformatted = []
                for elem in tiles:
                    tiles_reformatted.append([elem[0], elem[2], elem[1] - elem[0], elem[3] - elem[2]])

                other_tiles_reformatted = []
                for elem in other_tiles:
                    other_tiles_reformatted.append([elem[0], elem[2], elem[1] - elem[0], elem[3] - elem[2]])

                di.display_images([base_img, other_img], bboxes=[tiles_reformatted, other_tiles_reformatted])

            if len(additional_tiepoints) == 0:
                # if this happens, there is something wrong with the images,
                # so in this case it is better to stop tie point finding
                p.print_v("No additional tiepoints found", verbose=verbose, pbar=pbar)
                return None, None

            # stack tie points & confidences together
            additional_tiepoints = np.vstack(additional_tiepoints)
            additional_confidences = np.array(additional_confidences)

            # filter the tie-points again
            additional_tiepoints, additional_confidences, _ = filter_with_mask(additional_tiepoints,
                                                                               additional_confidences,
                                                                               mask_1, mask_2)

            p.print_v(f"{additional_tiepoints.shape[0]} additional tie-points found.", verbose=verbose, pbar=pbar)

            if keep_resized_points:
                additional_tiepoints = np.concatenate((additional_tiepoints, mkpts), axis=0)
                additional_confidences = np.concatenate((additional_confidences, mconf), axis=0)

            # remove duplicates
            additional_tiepoints, unique_indices = np.unique(additional_tiepoints, return_index=True, axis=0)
            additional_tiepoints = additional_tiepoints.astype(int)
            additional_confidences = additional_confidences[unique_indices]

            # apply threshold
            if min_threshold is not None:
                num_tp_before = additional_tiepoints.shape[0]
                additional_tiepoints = additional_tiepoints[additional_confidences >= min_threshold]
                additional_confidences = additional_confidences[additional_confidences >= min_threshold]
                num_tp_after = additional_tiepoints.shape[0]

                p.print_v(f"{num_tp_before - num_tp_after} tiepoints removed with a quality lower than "
                          f"{min_threshold}.", verbose=verbose, pbar=pbar)

            # filter for outliers
            if filter_outliers and additional_tiepoints.shape[0] >= 4:
                _, mask = cv2.findHomography(additional_tiepoints[:, 0:2], additional_tiepoints[:, 2:4],
                                             cv2.RANSAC, 5.0)
                mask = mask.flatten()

                p.print_v(f"{np.count_nonzero(mask)} outliers removed.", verbose=verbose, pbar=pbar)

                # 1 means outlier
                additional_tiepoints = additional_tiepoints[mask == 0]
                additional_confidences = additional_confidences[mask == 0]

            if debug_show_additional_tiepoints:
                dt.display_tiepoints([input_img1, input_img2],
                                     additional_tiepoints, additional_confidences,
                                     title=f"Additional tie points {additional_tiepoints.shape[0]}")

        else:
            additional_tiepoints = mkpts
            additional_confidences = mconf

        # if we already have many tie-points now we calculate a transformation matrix and do even more matching!
        if extra_matching and additional_tiepoints.shape[0] >= 3:

            p.print_v("Start extra matching:", verbose=verbose, pbar=pbar)

            # here we save all tiepoints
            extra_tiepoints = []
            extra_confidences = []

            switch_base_other = False

            # we need to iterate over the bigger image
            if img_1.shape[0] * img_1.shape[1] < img_2.shape[0] * img_2.shape[1]:
                switch_base_other = True

            if switch_base_other is False:
                base_img = img_1
                other_img = img_2
                base_mask = mask_1
                other_mask = mask_2
            else:
                base_img = img_2
                other_img = img_1
                base_mask = mask_2
                other_mask = mask_1

            # only the best tie-point should be used for the transformation
            min_confidence = 0.99
            while True:
                best_tie_points = additional_tiepoints[additional_confidences > min_confidence]
                best_confidences = additional_confidences[additional_confidences > min_confidence]
                if best_tie_points.shape[0] < 3:
                    min_confidence = min_confidence - 0.05
                else:
                    break

            if debug_show_transformation_points:
                dt.display_tiepoints([[input_img1, input_img2]], [best_tie_points], [best_confidences],
                                     title=f"Transformation tie points {best_tie_points.shape[0]} "
                                           f"with min confidence: {min_confidence}")

            # get affine transformation
            if switch_base_other is False:
                trans_mat = tf.estimate_transform('affine', best_tie_points[:, 0:2], best_tie_points[:, 2:4])
            else:
                trans_mat = tf.estimate_transform('affine', best_tie_points[:, 2:4], best_tie_points[:, 0:2])

            trans_mat = np.array(trans_mat)[0:2, :]  # noqa

            if debug_print_trans_mat:
                print(trans_mat)

            # how often does the maximum possible image fit in our base img
            num_x = math.ceil(base_img.shape[1] / max_width)
            num_y = math.ceil(base_img.shape[0] / max_height)

            # how much do we have too much
            too_much_x = num_x * max_width - base_img.shape[1]
            too_much_y = num_y * max_height - base_img.shape[0]

            # how much we need to shift the image every step
            if num_x - 1 == 0:
                reduce_x = 0
            else:
                reduce_x = int(too_much_x / (num_x - 1))
            if num_y - 1 == 0:
                reduce_y = 0
            else:
                reduce_y = int(too_much_y / (num_y - 1))

            # save tiles again for later displaying
            tiles = []
            other_tiles = []

            # init the progress bar
            max_counter = math.ceil(num_y) * math.ceil(num_x)
            add_counter = 0

            # iterate over the tiles again
            for y_counter in range(0, math.ceil(num_y)):
                for x_counter in range(0, math.ceil(num_x)):

                    add_counter += 1
                    p.print_v(f"Extra matching: {add_counter}/{max_counter}", verbose=verbose, pbar=pbar)

                    # calculate the extent of the tile
                    min_x = x_counter * max_width - reduce_x * x_counter
                    max_x = (x_counter + 1) * max_width - reduce_x * x_counter
                    min_y = y_counter * max_height - reduce_y * y_counter
                    max_y = (y_counter + 1) * max_height - reduce_y * y_counter

                    # create bounding box from extent
                    extent_points = np.asarray([
                        [min_x, min_y],
                        [min_x, max_y],
                        [max_x, max_y],
                        [max_x, min_y]
                    ])

                    # resample points
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        extent_points = np.hstack([extent_points, np.ones((extent_points.shape[0], 1))])
                        transformed_points = np.dot(extent_points, trans_mat.T)[:, :2].astype(int)

                    # get bounding box from resampled points
                    res_min_x = np.amin(transformed_points[:, 0])
                    res_max_x = np.amax(transformed_points[:, 0])
                    res_min_y = np.amin(transformed_points[:, 1])
                    res_max_y = np.amax(transformed_points[:, 1])

                    # some coords are out of range
                    if (res_min_x < 0 and res_max_x < 0) or (res_min_y < 0 and res_max_y < 0):
                        p.print_v("skip because res_x or res_y below zero", verbose=verbose, pbar=pbar)
                        continue

                    if (res_min_x > other_img.shape[1] and res_max_x > other_img.shape[1]) or \
                            (res_min_y > other_img.shape[0] and res_max_y > other_img.shape[0]):
                        p.print_v("skip because res_x or res_y bigger image shape", verbose=verbose, pbar=pbar)
                        continue

                    if (min_x < 0 and max_x < 0) or (min_y < 0 and max_y < 0):
                        p.print_v("skip because min_x or min_y below zero", verbose=verbose, pbar=pbar)
                        continue

                    if (min_x > base_img.shape[1] and max_x > base_img.shape[1]) or \
                            (min_y > base_img.shape[0] and max_y > base_img.shape[0]):
                        p.print_v("skip because min_x or min_y bigger image shape", verbose=verbose, pbar=pbar)
                        continue

                    min_x = max([0, min_x])
                    min_y = max([0, min_y])
                    max_x = min([base_img.shape[1], max_x])
                    max_y = min([base_img.shape[0], max_y])

                    res_min_x = max([0, res_min_x])
                    res_min_y = max([0, res_min_y])
                    res_max_x = min([other_img.shape[1], res_max_x])
                    res_max_y = min([other_img.shape[0], res_max_y])

                    # get the subsets for superglue extraction
                    base_subset = base_img[min_y:max_y, min_x:max_x]
                    other_subset = other_img[res_min_y:res_max_y, res_min_x:res_max_x]

                    # the shape must have a minimum size
                    if base_subset.shape[0] < 10 or base_subset.shape[1] < 10:
                        p.print_v("skip because base subset too small", verbose=verbose, pbar=pbar)
                        continue
                    if other_subset.shape[0] < 10 or other_subset.shape[1] < 10:
                        p.print_v("skip because other subset too small", verbose=verbose, pbar=pbar)
                        continue

                    # sometimes we don't need to look for tie-points here, because the boxes
                    # points are completely masked already
                    if base_mask is not None:
                        base_mask_subset = base_mask[min_y:max_y, min_x:max_x]
                        if np.sum(base_mask_subset) == 0:
                            p.print_v("skip because base subset is empty", verbose=verbose, pbar=pbar)
                            continue
                    if other_mask is not None:
                        other_mask_subset = other_mask[res_min_y:res_max_y, res_min_x:res_max_x]
                        if np.sum(other_mask_subset) == 0:
                            p.print_v("skip because other subset is empty", verbose=verbose, pbar=pbar)
                            continue

                    # save the tiles to display them later
                    tiles.append([min_x, max_x, min_y, max_y])
                    other_tiles.append([res_min_x, res_max_x, res_min_y, res_max_y])

                    # init the resize factors
                    sub_resize_factor_1, sub_resize_factor_2 = 1, 1

                    # get the tie points for these subsets
                    while True:
                        try:

                            if matching_method == "LightGlue":
                                mkpts1_sub, mkpts2_sub, mconf_sub = apply_lightglue(base_subset, other_subset)
                            elif matching_method == "SuperGlue":
                                mkpts1_sub, mkpts2_sub, mconf_sub = apply_superglue(base_subset, other_subset)

                            # that means we didn't raise an error but found a solution
                            break

                        except (Exception,) as e:
                            if "out of memory" in str(e):

                                # free the gpu
                                torch.cuda.empty_cache()

                                # calculate new height and width
                                new_base_width = int(0.9 * base_subset.shape[1])
                                new_base_height = int(0.9 * base_subset.shape[0])
                                new_other_width = int(0.9 * other_subset.shape[1])
                                new_other_height = int(0.9 * other_subset.shape[0])

                                # resize images again
                                _, base_subset, sub_resize_factor_1 = resize_img(base_subset, new_base_height,
                                                                                 new_base_width)
                                _, other_subset, sub_resize_factor_2 = resize_img(other_subset, new_other_height,
                                                                                  new_other_width)

                                p.print_v("Out of memory, new size ({base_subset.shape}, {other_subset.shape})",
                                          verbose=verbose, pbar=pbar)
                            else:
                                raise e

                    mkpts_sub = np.concatenate((mkpts1_sub, mkpts2_sub), axis=1)

                    if debug_show_extra_subset:

                        if y_counter == 4 and x_counter == 3:
                            dt.display_tiepoints([base_subset, other_subset], mkpts_sub, mconf_sub,
                                                 title=f"new tiepoints for subset ({y_counter}/{x_counter})")

                    mkpts_sub[:, 0] = mkpts_sub[:, 0] * 1 / sub_resize_factor_1
                    mkpts_sub[:, 1] = mkpts_sub[:, 1] * 1 / sub_resize_factor_1
                    mkpts_sub[:, 2] = mkpts_sub[:, 2] * 1 / sub_resize_factor_2
                    mkpts_sub[:, 3] = mkpts_sub[:, 3] * 1 / sub_resize_factor_2
                    mkpts_sub = mkpts_sub.astype(int)

                    # adapt the tie points of the other image to account that we look at a subset
                    mkpts_sub[:, 0] = mkpts_sub[:, 0] + min_x
                    mkpts_sub[:, 1] = mkpts_sub[:, 1] + min_y
                    mkpts_sub[:, 2] = mkpts_sub[:, 2] + res_min_x
                    mkpts_sub[:, 3] = mkpts_sub[:, 3] + res_min_y

                    # we need to switch the columns in tie points to match everything again
                    if switch_base_other:
                        temp = np.column_stack((mkpts_sub[:, 2], mkpts_sub[:, 3], mkpts_sub[:, 0], mkpts_sub[:, 1]))
                        mkpts_sub = temp

                    # add the tie points and confidence values to the list
                    extra_tiepoints.append(mkpts_sub)
                    for elem in mconf_sub:
                        extra_confidences.append(elem)

                    p.print_v(f"{mkpts_sub.shape[0]} tie-points matched", verbose=verbose, pbar=pbar)

            p.print_v("Extra matching finished", verbose=verbose, pbar=pbar)

            if debug_show_extra_tiles:

                tiles_reformatted = []
                for elem in tiles:
                    tiles_reformatted.append([elem[0], elem[2], elem[1] - elem[0], elem[3] - elem[2]])

                other_tiles_reformatted = []
                for elem in other_tiles:
                    other_tiles_reformatted.append([elem[0], elem[2], elem[1] - elem[0], elem[3] - elem[2]])

                di.display_images([base_img, other_img], bboxes=[tiles_reformatted, other_tiles_reformatted])

            # stack tie points & confidences together
            if len(extra_tiepoints) > 0:
                extra_tiepoints = np.vstack(extra_tiepoints)
                extra_confidences = np.array(extra_confidences)
            else:
                extra_tiepoints = np.zeros((0, 4))
                extra_confidences = []

            # mask tie_points
            extra_tiepoints, extra_confidences, extra_indices = filter_with_mask(extra_tiepoints, extra_confidences,
                                                                                 mask_1, mask_2)

            p.print_v(f"{extra_tiepoints.shape[0]} extra tie-points found.", verbose=verbose, pbar=pbar)

            if keep_additional_points:
                extra_tiepoints = np.concatenate((extra_tiepoints, additional_tiepoints))
                extra_confidences = np.concatenate((extra_confidences, additional_confidences))

            # remove duplicates
            extra_tiepoints, unique_indices = np.unique(extra_tiepoints, return_index=True, axis=0)
            extra_tiepoints = extra_tiepoints.astype(int)
            extra_confidences = extra_confidences[unique_indices]

            # apply threshold
            if min_threshold is not None:
                num_tp_before = extra_tiepoints.shape[0]
                extra_tiepoints = extra_tiepoints[extra_confidences >= min_threshold]
                extra_confidences = extra_confidences[extra_confidences >= min_threshold]
                num_tp_after = extra_tiepoints.shape[0]

                p.print_v(f"{num_tp_before - num_tp_after} tiepoints removed with a quality lower than "
                          f"{min_threshold}.", verbose=verbose, pbar=pbar)

            # filter for outliers
            if filter_outliers and extra_tiepoints.shape[0] >= 4:
                _, mask = cv2.findHomography(extra_tiepoints[:, 0:2], extra_tiepoints[:, 2:4], cv2.RANSAC, 5.0)  # noqa
                mask = mask.flatten()

                p.print_v(f"{np.count_nonzero(mask)} outliers removed.", verbose=verbose, pbar=pbar)

                # 1 means outlier
                extra_tiepoints = extra_tiepoints[mask == 0]
                extra_confidences = extra_confidences[mask == 0]
        else:
            extra_tiepoints = additional_tiepoints
            extra_confidences = additional_confidences

        if extra_tiepoints.shape[0] == 0:
            p.print_v("No extra tiepoints found", verbose=verbose, pbar=pbar)
        else:
            p.print_v(f"In total {extra_tiepoints.shape[0]} tie-points were found "
                      f"with a quality of {round(np.average(extra_confidences), 3)}", verbose=verbose, pbar=pbar)

        if debug_clean_tie_points and extra_tiepoints.shape[0] > 0:

            def average_rows(data, confs, threshold=10):
                """Check and average rows based on the first two columns."""
                unique, indices, counts = np.unique(data[:, :2], axis=0, return_inverse=True, return_counts=True)
                new_data = []
                new_confs = []

                for idx, point in enumerate(unique):
                    same_points = data[indices == idx]
                    same_confs = np.array(confs)[indices == idx]

                    # If there is only one such point or all points within the threshold
                    if counts[idx] == 1 or np.all(
                            np.linalg.norm(same_points[:, 2:4] - same_points[0, 2:4], axis=1) <= threshold):
                        new_data.append(np.concatenate([point, np.mean(same_points[:, 2:4], axis=0)]))
                        new_confs.append(np.mean(same_confs))
                    else:
                        # Removing all points that exceed the threshold
                        pass

                return np.array(new_data), new_confs

            extra_tiepoints, extra_confidences = average_rows(extra_tiepoints, extra_confidences, clean_threshold)

            if extra_tiepoints.shape[0] == 0:
                return None, None

            # Step 3: For Image 2 (swap columns, and process like Image 1)
            swapped_tiepoints = extra_tiepoints[:, [2, 3, 0, 1]]
            swapped_tiepoints, extra_confidences = average_rows(swapped_tiepoints, extra_confidences, clean_threshold)

            if swapped_tiepoints.shape[0] == 0:
                return None, None

            extra_tiepoints = swapped_tiepoints[:, [2, 3, 0, 1]]
            extra_tiepoints = extra_tiepoints.astype(int)
            extra_confidences = np.asarray(extra_confidences)

            p.print_v(f"There are {extra_tiepoints.shape[0]} tie-points left after cleaning", verbose=verbose,
                      pbar=pbar)

        if debug_show_final_tiepoints:
            dt.display_tiepoints([input_img1, input_img2], extra_tiepoints, extra_confidences,
                                 title=f"Final tie points {extra_tiepoints.shape[0]}")
    except (Exception,) as e:
        if catch:
            p.print_v("Failed: find_tie_points", verbose, pbar=pbar)
            return None, None
        else:
            raise e

    p.print_v("Finished: find_tie_points", verbose, pbar=pbar)

    return extra_tiepoints, extra_confidences


if __name__ == "__main__":
    img_id1 = "CA216632V0282"
    img_id2 = "CA216632V0283"

    import base.load_image_from_file as liff

    img1 = liff.load_image_from_file(img_id1)
    img2 = liff.load_image_from_file(img_id2)

    mask1 = np.zeros_like(img1)
    mask2 = np.zeros_like(img2)

    import base.remove_borders as rb

    _, edge_dims1 = rb.remove_borders(img1, img_id1, return_edge_dims=True)
    mask1[edge_dims1[2]:edge_dims1[3], edge_dims1[0]:edge_dims1[1]] = 1

    _, edge_dims2 = rb.remove_borders(img2, img_id2, return_edge_dims=True)
    mask2[edge_dims2[2]:edge_dims2[3], edge_dims2[0]:edge_dims2[1]] = 1

    points, conf = find_tie_points(img1, img2,
                                   mask_1=mask1, mask_2=mask2,
                                   extra_matching=True,
                                   catch=False, verbose=True)
