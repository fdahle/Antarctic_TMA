import cv2 #in order for some cv methods
import os

def load_image(imagePath, get_path=None, subset=None):

    scriptPath = os.path.realpath(__file__)
    projectFolder = scriptPath[:-26]

    absoluteImagePath = projectFolder + "/" +  imagePath

    img = cv2.imread(absoluteImagePath)

    #convert to grayscale if needed
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if get_path:
        return img, absoluteImagePath
    else:
        return img
