
from tqdm import tqdm

import src.load.load_image as li

import src.georef.georef_sat as gs
import src.georef.georef_img as gi
import src.georef.georef_calc as gc

def georef():

    # init the geo-reference objects
    georefSat = gs.GeorefSatellite()
    georefImg = gi.GeorefImage()
    georefCalc = gc.GeorefCalc()


    # iterate images
    for image_id in input_ids:

        # load the image
        image = li.load_image(image_id)



if __name__ == "__main__":


