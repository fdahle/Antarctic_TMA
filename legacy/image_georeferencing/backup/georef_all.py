
from tqdm import tqdm

#import image_georeferencing.georef_img as gi
import image_georeferencing.old.georef_sat as gs
import image_georeferencing.derive_image_positions as dip

import base.connect_to_db as ctd

def georef_all():

    # get all unreferenced images
    sql_string = "SELECT image_id FROM images_extracted WHERE footprint IS NULL " \
                 "OR footprint_type != 'satellite'"
    data = ctd.get_data_from_db(sql_string)
    image_ids = data['image_id'].values.tolist()

    # how many images did we georeference based on satellite
    nr_georef_sat = 0
    nr_failed = 0
    nr_try_again = 0

    lst_georeferenced = []
    lst_failed = []
    lst_try_again = []

    # first we try to geo-reference all images with other satellite images
    print("Geo-reference with satellite images")
    for image_id in tqdm(image_ids):

        status = gs.georef_sat([image_id])

        # best outcome --> image is geo-referenced
        if status == "georeferenced":
            nr_georef_sat += 1
            lst_georeferenced.append(image_id)

        # worst outcome --> there's an error
        elif status == "failed":
            nr_failed += 1
            lst_georeferenced.append(image_id)

        # we have too few tie-points or these are not good --> we need to try again later
        elif status == "invalid" or status == "too_few_tps":
            nr_try_again += 1
            lst_try_again.append(image_id)

    # now that we have already a bunch of geo-referenced images we can estimate the positions of images
    for image_id in tqdm(lst_try_again):

        # get an estimated position and approx_footprint
        position_estimated, footprint_estimated = dip.derive_image_position(image_id)

        # try again with estimated position
        gs.georef_sat(image_id)


    exit()



        #gi.georef_img()

if __name__ == "__main__":

    georef_all()