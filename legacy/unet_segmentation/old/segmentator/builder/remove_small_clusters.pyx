import numpy as np

#for image processing
from scipy import ndimage as ndi
from skimage.segmentation import watershed, expand_labels

def remove_small_clusters(img, minClusterSize):

    #filled = np.copy(img)
    filled = np.zeros_like(img)

    #cluster the image
    uv = np.unique(img)
    s = ndi.generate_binary_structure(2,2)
    cum_num = 0

    clustered = np.zeros_like(img)
    for v in uv:
        labeled_array, num_features = ndi.label((img==v).astype(int), structure=s)
        clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
        cum_num += num_features

    #count the pixels of each cluster and put it in a list
    unique, counts = np.unique(clustered, return_counts=True)
    clusters = np.column_stack([unique, counts])

    #iterate all clusters and set clusters below the threshold to background
    i = 101
    for elem in clusters:
        if elem[1] > minClusterSize:
            filled[clustered == elem[0]] = i
            i+=1

    #fill the background with the surrounding pixels
    filled = expand_labels(filled, distance=1000)

    #return the new image
    return filled
