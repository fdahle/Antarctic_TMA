import copy
import cv2
import glob
import json
import math
import numpy as np
import os
import torch
import warnings

import base.print_v as p
import base.resize_image as ri

import unet_segmentation.u_net as u_net
import unet_segmentation.u_net_small as u_net_small


def segment_image(input_img, image_id=None,
                  model_id="LATEST", model_path=None, model_type="normal",
                  bool_normalize=None,
                  resize_images=None, new_size_x=None, new_size_y=None,
                  apply_threshold=None, min_threshold=None,
                  num_layers=None, num_output_layers=None,
                  catch=True, verbose=False, pbar=None):

    """
    segment_image(input_img, image_id, model_id, model:path, resize_images, new_size, apply_threshold, min_threshold,
                  num_layers, num_output_layers, verbose):
    This function applies semantic segmentation on an image based on a UNET model.
    Args:
        input_img (np-array): The aerial image that should be segmented
        image_id (String): The image_id of the image, not required, will only be used for printing if verbose=True
        model_id (String) The image_id of the model we want to use for segmenting. Note that it is possible
            to use the keywords "LATEST", "LATEST_PTH" or "LATEST_PT". These allow to use either the newest model
            in the folder, or respective the newest *.pth or *.pt model.
        model_path (String): The path where the model file is located. If this value is none, a default model
            path specified in the beginning of this function is used.
        model_type (String): This specifies if the normal U-net with 4 layers ("normal") or the small
            U-net with 3 layers ("small") should be used.
        bool_normalize (Boolean): If true, the image will be normalized before segmenting
        resize_images (Boolean): It is possible to resize the images before they are segmented in order
            to increase the speed
        new_size_x (int): the x-size for the resized images
        new_size_y (int): the y-size for the resized images
        apply_threshold (Boolean): If this is true, the probability values for every pixel will be checked.
            If this value is lower than the value specified in 'min_threshold', this pixel will be assigned the class
            'unknown'
        min_threshold (float): The minimum threshold value for 'apply_threshold'
        num_layers (int): The number of input layers the model has
        num_output_layers (int): The number of output layers the model has
        catch (Boolean, True): If true and something is going wrong (for example no fid points),
            the operation will continue and not crash
        verbose (Boolean, False): If true, the status of the operations are printed
        pbar (tqdm-progress-bar): If this is true, the text output will be not shown as text, but
            as a description in a tqdm-progress-bar
    Returns:
        pred (np-array): The segmented input image with the same dimensions
        probabilities (np-array): The probabilities for each pixel and for each output class.
            (Num output classes x height x width)
        highest_prob (np-array): The image with each pixel having the highest probability value from the six classes
            (height x width)
        model_name (String): The name of the model used for segmentation
    """

    # the segmentation has the following classes:
    # 1: ice, 2: snow, 3: rocks, 4: water, 5: clouds, 6:sky, 7: unknown

    p.print_v(f"Start: segment_image ({image_id})", verbose=verbose, pbar=pbar)

    # load the json to get default values
    json_folder = os.path.dirname(os.path.realpath(__file__))
    with open(json_folder + "/params.json") as j_file:
        json_data = json.load(j_file)

    if model_path is None:
        model_path = json_data["path_folder_segmentation_model"]

    if bool_normalize is None:
        bool_normalize = json_data["segment_image_bool_normalize"]

    if resize_images is None:
        resize_images = json_data["segment_image_resize_images"]

    if new_size_x is None:
        new_size_x = json_data["segment_image_new_size_x"]

    if new_size_y is None:
        new_size_y = json_data["segment_image_new_size_y"]

    if apply_threshold is None:
        apply_threshold = json_data["segment_image_apply_threshold"]

    if min_threshold is None:
        min_threshold = json_data["segment_image_min_threshold"]

    if num_layers is None:
        num_layers = json_data["segment_image_num_layers"]

    if num_output_layers is None:
        num_output_layers = json_data["segment_image_num_output_layers"]

    # set new size var
    new_size = (new_size_x, new_size_y)

    # deep copy to not change the original
    img = copy.deepcopy(input_img)

    try:
        # resize image if wished (to increase the speed)
        if resize_images:
            orig_shape = img.shape
            img = ri.resize_image(img, (new_size[1], new_size[0]))
        else:
            # so that pycharm is not complaining
            orig_shape = None

        # get absolute model path
        if model_id == "LATEST":

            list_of_files_pt = glob.glob(model_path + "/*.pt")
            list_of_files_pth = glob.glob(model_path + "/*.pth")

            list_of_files = list_of_files_pt + list_of_files_pth

            if len(list_of_files) == 0:
                print(f"segment_images (latest_pt): No model could be found at {model_path}")
                exit()

            latest_model = max(list_of_files, key=os.path.getmtime)
            absolute_model_path = latest_model

        elif model_id == "LATEST_PT":
            list_of_files = glob.glob(model_path + "/*.pt")

            if len(list_of_files) == 0:
                print(f"segment_images (latest_pt): No model could be found at {model_path}")
                exit()

            latest_model = max(list_of_files, key=os.path.getmtime)
            absolute_model_path = latest_model

        elif model_id == "LATEST_PTH":
            list_of_files = glob.glob(model_path + "/*.pth")

            if len(list_of_files) == 0:
                print("segment_images (latest_pth): No model could be found at {}".format(model_path))
                exit()

            latest_model = max(list_of_files, key=os.path.getmtime)
            absolute_model_path = latest_model

        else:
            absolute_model_path = model_path + "/" + model_id
            if "." in absolute_model_path:
                pass
            elif os.path.exists(absolute_model_path + ".pt"):
                absolute_model_path = absolute_model_path + ".pt"
            elif os.path.exists(absolute_model_path + ".pth"):
                absolute_model_path = absolute_model_path + ".pth"
            else:
                print("Model {} is not existing at {}".format(model_id, absolute_model_path))
                exit()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            # set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # load the model
            if absolute_model_path.split(".")[-1] == "pt":
                model = torch.load(absolute_model_path)
                model.eval()
            elif absolute_model_path.split(".")[-1] == "pth":

                if model_type == "small":
                    model = u_net_small.UNET_SMALL(num_layers, num_output_layers)
                elif model_type == "normal":
                    model = u_net.UNET(num_layers, num_output_layers)
                model.to(device)

                checkpoint = torch.load(absolute_model_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
            else:
                model = None
                print("This is not a valid file-type")
                exit()

        p.print_v(f"Segmentation for {image_id} is applied with model"
                  f" {absolute_model_path.split('/')[-1]}",
                  verbose=verbose, pbar=pbar)

        # get the name of the model
        model_name = absolute_model_path.split('/')[-1]

        if bool_normalize:
            # normalize image
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # convert image so that it can be predicted
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = img.float()
        img = img.to(device)

        # the actual prediction 6 layers with prob for each layer
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            probabilities = model(img)

        probabilities = probabilities.cpu().detach().numpy()
        probabilities = probabilities[0]

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        sigmoid_v = np.vectorize(sigmoid)
        probabilities = sigmoid_v(probabilities)

        # np.set_printoptions(formatter={'all': lambda x: str(x)})

        # 1 layer: per pixel the class with the highest prob
        pred = np.argmax(probabilities, axis=0)

        # 1 layer: per pixel the highest prob
        highest_prob = np.amax(probabilities, axis=0)

        if apply_threshold:
            pred[highest_prob < min_threshold] = 6

        # resize back to original
        if resize_images:
            _temp = np.empty((probabilities.shape[0], orig_shape[0], orig_shape[1]))
            for i, elem in enumerate(probabilities):
                _temp[i] = cv2.resize(elem, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            probabilities = _temp

            pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            highest_prob = cv2.resize(highest_prob, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

        # pred to u int 8 to save some memory
        pred = pred.astype(np.uint8)

        # increment prediction so that the colors matches
        pred = pred + 1

    except (Exception,) as e:
        if catch:
            p.print_v(f"Failed: segment_image ({image_id})", verbose=verbose, pbar=pbar)
            return None, None, None, None
        else:
            raise e

    p.print_v(f"Finished: segment_image ({image_id})", verbose=verbose, pbar=pbar)

    return pred, probabilities, highest_prob, model_name
