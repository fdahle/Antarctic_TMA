import json
import os

with open('../params1.json') as json_file:
    params1 = json.load(json_file)

data_folder = "../../../data/"

segmentFolder = data_folder + "aerial/TMA/segmented/unsupervised"
outputFolder = data_folder + "aerial/TMA/segmented/supervised"

for filename in os.listdir(outputFolder):
    if filename.endswith(".tif"):

        if os.path.isfile(segmentFolder + "/" + filename):
            os.replace(segmentFolder + "/" + filename, segmentFolder + "/done/" + filename)
            print(filename, "moved")
