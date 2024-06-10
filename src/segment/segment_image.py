# Package imports
import copy
import numpy as np
import torch


def segment_image(image, normalize=False):

    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    return probabilities