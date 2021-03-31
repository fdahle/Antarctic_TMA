import subprocess #to execute shell commands
import os
import sys

overwrite = False
scan_resolution = 0.025

imageFolder = "../../data/images"
originalImages = imageFolder + "/images"
xmlFiles = imageFolder + "/images/Ori-InterneScan"
resampledImages = imageFolder + "/resampled"

images = []

#get all images
for filename in os.listdir(originalImages):
    if filename.endswith(".tif"):
        images.append(filename)

#remove the images already done
if overwrite == False:
    for filename in os.listdir(resampledImages):
        if filename.endswith(".tif"):
            images.remove(filename)

#only get the images with a xml file
images_to_resample = []
for xmlFile in os.listdir(xmlImages):
    if xmlFile.endswith("xml"):
        filename = xmlFile.split("-")[1].split(".")[0] + ".tif"
        if filename in images:
            images_to_resample.append(filename)

#change to the image folder
absImgFolder = os.path.abspath(originalImages)
subprocess.call('cd ' + absImgFolder, shell=True)

for img in images_to_resample:


    shellString = 'mm3d ReSampFid "' + img + '" ' + str(scan_resolution)
    print(shellString)

    subprocess.call(shellString, shell=True)
