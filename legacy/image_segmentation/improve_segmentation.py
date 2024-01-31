import copy
import cv2
import json
import numpy as np
import os

from scipy import ndimage as ndi
from skimage.segmentation import expand_labels

import base.print_v as p
import base.resize_image as ri

import display.display_images as di

debug_show_images = False


def improve_segmentation(image_id, input_segmented, input_probabilities,
                         improvement_methods=None,
                         resize_image=None, resize_size_x=None, resize_size_y=None,
                         catch=True, verbose=False, pbar=None):

    """improve_segmentation(input_segmented, input_probabilities, image_id, catch, verbose):
    This function takes a segmented images and applies a set of different rules in order to improve the segmentations.
    Input_probabilities and image_id are not required, but can improve the correction even more.
    Args:
        input_segmented (np-array): The unprocessed segmented image
        input_probabilities (np-array): The probabilities for each class
        improvement_methods (list): This list contains all improvement methods that will be applied
        resize_image (Boolean): If true we will resize the image before improving and
            resize back afterwards
        resize_size_x (int): The x-size we want to resize to
        resize_size_y (int): The y-size we want to resize to
        image_id (String): The image image_id of the segmented image
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        segmented (np-array): The corrected segmented image
        """

    p.print_v(f"Start: improve_segmentation ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if improvement_methods is None:
        improvement_methods = json_data["improve_segmented_image_improvement_methods"]

    if resize_image is None:
        resize_image = json_data["improve_segmented_image_resize_image"]

    if resize_size_x is None:
        resize_size_x = json_data["improve_segmented_image_resize_size_x"]

    if resize_size_y is None:
        resize_size_y = json_data["improve_segmented_image_resize_size_y"]

    assert type(input_segmented) is np.ndarray, \
        "The segmented file is not a np-array"

    # deep copy to not change the original
    segmented = copy.deepcopy(input_segmented)

    # deep copy of the probabilities
    if input_probabilities is not None:
        probabilities = copy.deepcopy(input_probabilities)
    else:
        # so that the ide is not complaining
        probabilities = None

    try:

        # resize images (so that the improving is faster)
        if resize_image:
            orig_shape = segmented.shape
            new_size = (resize_size_y, resize_size_x)
            segmented = ri.resize_image(segmented, new_size)

        # blur the image so that the
        segmented = cv2.medianBlur(segmented, 7)

        if debug_show_images:
            di.display_images(segmented, title="blurred")

        if resize_image:
            probabilities_new = []
            new_size = (resize_size_y, resize_size_x)
            for i, elem in enumerate(probabilities):
                probabilities_new.append(ri.resize_image(elem, new_size))
                probabilities = np.asarray(probabilities_new)

        # remove sky from images with V in filename
        def remove_sky(_segmented):

            _segmented[_segmented == 6] = 0
            _segmented = expand_labels(_segmented, distance=1000)

            return _segmented

        if "remove_sky" in improvement_methods:
            if "V" in image_id:
                p.print_v("Runtime: improve_segmentation (remove sky segments)", verbose=verbose, pbar=pbar)
                segmented = remove_sky(segmented)

            if debug_show_images:
                di.display_images(segmented, title="remove sky segments")

        # if there are patches with the class unknown, these will be filled
        def fill_unknown(_segmented, probs):
            _segmented[_segmented == 7] = np.amax(probs, axis=0)[_segmented == 7]
            return _segmented

        if "fill_unknown" in improvement_methods:
            if input_probabilities is not None:
                p.print_v("Runtime: improve_segmentation(fill unknown segments)", verbose=verbose, pbar=pbar)
                segmented = fill_unknown(segmented, probabilities)

            if debug_show_images:
                di.display_images(segmented, title="fill unknown segments")

        # remove stuff in sky that should not be there
        def remove_clusters_in_sky(_segmented):
            min_perc = 20

            for j in range(_segmented.shape[0]):
                uniques, counts = np.unique(_segmented[j, :], return_counts=True)
                percentages = dict(zip(uniques, counts * 100 / len(segmented[j, :])))
                try:
                    if percentages[6] > min_perc:
                        _segmented[0:j, :] = 6
                except (Exception,):
                    pass

                if j % 100 == 0:
                    min_perc = min_perc + 5

            binary = copy.deepcopy(segmented)

            # make binary for sky and non sky
            binary[binary != 6] = 0
            binary[binary == 6] = 1

            # cluster the image
            uv = np.unique(binary)
            s = ndi.generate_binary_structure(2, 2)
            cum_num = 0

            clustered = np.zeros_like(binary)
            for v in uv:
                labeled_array, num_features = ndi.label((binary == v).astype(int), structure=s)
                clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
                cum_num += num_features

            # count the pixels of each cluster and put it in a list
            unique, counts = np.unique(clustered, return_counts=True)
            clusters = np.column_stack([unique, counts])

            # sort clusters
            clusters_sorted = clusters[np.argsort(clusters[:, 1])]

            filled = np.ones_like(binary)

            # iterate all clusters and set clusters below the threshold to background
            for _elem in clusters_sorted:
                if _elem[1] < 50000:

                    filled[clustered == _elem[0]] = 0
                else:
                    break

            filled[filled != 0] = _segmented[filled != 0]

            # fill the background with the surrounding pixels
            filled = expand_labels(filled, distance=1000)

            return filled

        if "remove_clusters_in_sky" in improvement_methods:
            if "V" not in image_id:
                p.print_v("Runtime: improve_segmentation (remove clusters in sky)", verbose=verbose, pbar=pbar)
                segmented = remove_clusters_in_sky(segmented)

                if debug_show_images:
                    di.display_images(segmented, title="remove clusters in sky")

        # enlarge sky to a complete row if bigger than threshold
        def enlarge_sky(_segmented):

            min_perc = 20

            for j in range(_segmented.shape[0]):
                uniques, counts = np.unique(_segmented[j, :], return_counts=True)
                percentages = dict(zip(uniques, counts * 100 / len(_segmented[j, :])))
                try:
                    if percentages[6] > min_perc:
                        _segmented[0:j, :] = 6
                except (Exception,):
                    pass

                if j % 100 == 0:
                    min_perc = min_perc + 5

            return _segmented

        if "enlarge_sky" in improvement_methods:
            if "V" not in image_id:
                p.print_v("Runtime: improve_segmentation (enlarge sky)", verbose=verbose, pbar=pbar)
                segmented = enlarge_sky(segmented)

                if debug_show_images:
                    di.display_images(segmented, title="enlarge sky")

        # remove very small patches
        def remove_small_patches(_segmented):

            temp = copy.deepcopy(_segmented)

            filled = np.zeros_like(temp)

            # cluster the image
            uv = np.unique(temp)
            s = ndi.generate_binary_structure(2, 2)
            cum_num = 0

            clustered = np.zeros_like(temp).astype(np.uint16)
            for v in uv:
                labeled_array, num_features = ndi.label((temp == v).astype(int), structure=s)
                clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
                cum_num += num_features

            # count the pixels of each cluster and put it in a list
            unique, counts = np.unique(clustered, return_counts=True)
            clusters = np.column_stack([unique, counts])

            # sort clusters
            clusters_sorted = np.flip(clusters[np.argsort(clusters[:, 1])], axis=0)

            # iterate all clusters and set clusters below the threshold to background
            for _elem in clusters_sorted:
                if _elem[1] > 50:

                    # get the class
                    class_idx = np.where(clustered == _elem[0])
                    class_idx = (class_idx[0][0], class_idx[1][0])
                    elem_class = temp[class_idx]

                    filled[clustered == _elem[0]] = elem_class
                else:
                    break

            # fill the background with the surrounding pixels
            filled = expand_labels(filled, distance=1000)

            # return the new image
            return filled

        if "remove_small_patches" in improvement_methods:
            p.print_v("Runtime: improve_segmentation (remove small patches)", verbose=verbose, pbar=pbar)
            segmented = remove_small_patches(segmented)

            if debug_show_images:
                di.display_images(segmented, title="remove small patches")

        # remove the sky in the ground
        def remove_sky(_segmented):

            binary = copy.deepcopy(segmented)

            # make binary for sky and non sky
            binary[binary != 6] = 0
            binary[binary == 6] = 1

            if np.sum(binary) == 0:
                return segmented

            # cluster the image
            uv = np.unique(binary)
            s = ndi.generate_binary_structure(2, 2)
            cum_num = 0

            clustered = np.zeros_like(binary).astype(np.uint16)
            for v in uv:
                labeled_array, num_features = ndi.label((binary == v).astype(int), structure=s)
                clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
                cum_num += num_features

            # count the pixels of each cluster and put it in a list
            unique, counts = np.unique(clustered, return_counts=True)
            clusters = np.column_stack([unique, counts])

            # remove non sky element in clusters
            clusters_to_keep = []
            for _elem in clusters:
                pos = np.where(clustered == _elem[0])
                binary_id = binary[pos[0][0], pos[1][0]]
                if binary_id == 1:
                    clusters_to_keep.append(_elem)
            clusters_to_keep = np.asarray(clusters_to_keep)

            # sort clusters and remove the biggest
            clusters_sorted = clusters_to_keep[np.argsort(clusters_to_keep[:, 1])]
            clusters_sorted = clusters_sorted[:-1]

            if len(clusters_sorted) == 0:
                return _segmented

            # set the other cluster elements to background
            for _elem in clusters_sorted:
                _segmented[clustered == _elem[0]] = 0

            # fill the background with the surrounding pixels
            _segmented = expand_labels(_segmented, distance=1000)

            return _segmented

        if "remove_sky_ground" in improvement_methods:
            p.print_v("Runtime: improve_segmentation (remove sky in the ground)", verbose=verbose, pbar=pbar)
            segmented = remove_sky(segmented)

            if debug_show_images:
                di.display_images(segmented, title="remove sky in the ground")

        def check_logical_patches(_segmented, probs):

            temp = copy.deepcopy(_segmented)

            # cluster the image
            uv = np.unique(temp)
            s = ndi.generate_binary_structure(2, 2)
            cum_num = 0

            clustered = np.zeros_like(temp)
            for v in uv:
                labeled_array, num_features = ndi.label((temp == v).astype(int), structure=s)
                clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
                cum_num += num_features

            # count the pixels of each cluster and put it in a list
            unique, counts = np.unique(clustered, return_counts=True)
            clusters = np.column_stack([unique, counts])

            # get neighbouring segments
            # centers = np.array([np.mean(np.nonzero(clustered == j), axis=1) for j in unique])
            vs_right = np.vstack([clustered[:, :-1].ravel(), clustered[:, 1:].ravel()])
            vs_below = np.vstack([clustered[:-1, :].ravel(), clustered[1:, :].ravel()])
            b_neighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1).T

            # delete neighbours with themselves
            b_neighbors = np.delete(b_neighbors, np.where(b_neighbors[:, 0] == b_neighbors[:, 1]), axis=0)

            for _elem in clusters:

                segment_id = _elem[0]
                orig_class = int(np.median(_segmented[clustered == segment_id]))

                # get neighbours of this segment
                neighbours = b_neighbors[b_neighbors[:, 0] == segment_id]

                neighbour_classes = []

                if len(neighbours) == 0:
                    continue

                # iterate neighbours and get their classes
                for neighbour in neighbours[0]:
                    neighbour_class = int(np.median(_segmented[clustered == neighbour]))

                    neighbour_classes.append(neighbour_class)

                avg_pred_vals = {}

                # iterate all predictions
                for j, prediction in enumerate(probs):
                    avg_pred = np.average(prediction[clustered == segment_id])
                    avg_pred_vals[j] = avg_pred

                # water to rock/snow
                if orig_class == 4:

                    if avg_pred_vals[3] > 0.75 and 4 not in neighbours:

                        _segmented[clustered == segment_id] = 3
                        p.print_v("Runtime: improve_segmentation (check logical patches: Water to rocks)",
                                  verbose=verbose, pbar=pbar)
                    elif avg_pred_vals[4] - avg_pred_vals[2] < 0.001 and 4 not in neighbours:
                        _segmented[clustered == segment_id] = 2
                        p.print_v("Runtime: improve_segmentation (check logical patches: Water to snow)",
                                  verbose=verbose, pbar=pbar)

                # clouds to snow
                if orig_class == 5:
                    if avg_pred_vals[5] - avg_pred_vals[2] < 0.01 and 5 not in neighbours:
                        _segmented[clustered == segment_id] = 2
                        p.print_v("Runtime: improve_segmentation (check logical patches: Clouds to snow)",
                                  verbose=verbose, pbar=pbar)

                if orig_class == 2:
                    if avg_pred_vals[5] > 0.9 and 5 in neighbours:
                        _segmented[clustered == segment_id] = 5
                        p.print_v("Runtime: improve_segmentation (check logical patches: Snow to clouds)",
                                  verbose=verbose, pbar=pbar)

            return _segmented

        if "check_logical_patches" in improvement_methods:
            if input_probabilities is not None:
                p.print_v("Runtime: improve_segmentation (check logical patches)",
                          verbose=verbose, pbar=pbar)
                segmented = check_logical_patches(segmented, probabilities)

            if debug_show_images:
                di.display_images(segmented, title="check logical patches")

        # resize images back to original size
        if resize_image:
            segmented = ri.resize_image(segmented, (orig_shape[0], orig_shape[1]))  # noqa

    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: improve_segmentation ({image_id})", verbose=verbose, pbar=pbar)
            return None
        else:
            raise e

    p.print_v(f"Finished: improve_segmentation ({image_id})", verbose=verbose, pbar=pbar)

    return segmented
