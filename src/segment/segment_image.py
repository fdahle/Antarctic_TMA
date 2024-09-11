# Library imports
import copy
import numpy as np
import torch

import src.segment.classes.u_net as u_net

# CONSTANTS
MODEL_PATH = "/data/ATM/data_1/machine_learning/semantic_segmentation/test_cross_noise_360.pth"

def segment_image(image, normalize=False,
                  num_layers=1, num_output_layers=6,
                  min_threshold=None,
                  return_model_name=False):

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # load the model
    if MODEL_PATH.split(".")[-1] == "pt":
        model = torch.load(MODEL_PATH)
        model.eval()
    elif MODEL_PATH.split(".")[-1] == "pth":

        model = u_net.UNET(num_layers, num_output_layers)
        model.to(device)
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
    else:
        print("This is not a valid file-type")
        exit()

    # copy image to avoid changing the original image
    image = copy.deepcopy(image)

    if normalize:
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # prepare image for segmentation
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).float()
    image = image.to(device)

    # get the predictions for each layer
    probabilities = model(image)

    # convert probs back to numpy
    probabilities = probabilities.cpu().detach().numpy()
    probabilities = probabilities[0]

    # get per pixel the class with the highest prob
    pred = np.argmax(probabilities, axis=0)

    # get per pixel the highest prob
    highest_prob = np.amax(probabilities, axis=0)

    if min_threshold is not None:
        pred[highest_prob < min_threshold] = 6

    # pred to u int 8 to save some memory
    pred = pred.astype(np.uint8)

    # add 1 to the prediction to match the classes
    pred = pred + 1

    if return_model_name:
        # get model name
        model_name = MODEL_PATH.split("/")[-1]
        return pred, probabilities, model_name
    else:
        return pred, probabilities
