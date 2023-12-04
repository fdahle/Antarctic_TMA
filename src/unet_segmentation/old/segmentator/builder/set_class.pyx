import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import copy

def set_class(imgSegmented,imgOutput,x,y,segmentId,clustered,bool_scribbled, bool_box_drawn, overwrite):

  imgOutput = copy.deepcopy(imgOutput)

  if clustered is None or bool_scribbled or bool_box_drawn:
    #cluster the image
    uv = np.unique(imgSegmented)
    s = ndi.generate_binary_structure(2,2)
    cum_num = 0
    clustered = np.zeros_like(imgSegmented)
    for v in uv[1:]:
        labeled_array, num_features = ndi.label((imgSegmented==v).astype(int), structure=s)
        clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
        cum_num += num_features

  #get the cluster from the image
  clusterId = clustered[y,x]
  selected_class = imgOutput[y,x]


  if overwrite:
    imgOutput[clustered == clusterId] = segmentId
  else:
    imgOutput[(clustered == clusterId) & (imgOutput==0)] = segmentId

  return imgOutput, clustered
