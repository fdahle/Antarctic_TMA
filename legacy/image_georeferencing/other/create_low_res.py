import os
import shapely.wkt

from tqdm import tqdm

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.remove_borders as rb
import base.resize_image as ri

import image_georeferencing.georef_calc as gc

overwrite = True

types = ["sat", "sat_est", "img", "calc"]

def create_low_res():

    fld = "/data_1/ATM/data_1/playground/georef4/low_res/images"

    sql_string = "SELECT images.image_id, images.azimuth, " \
                 "ST_AsText(images_extracted.footprint_exact) AS footprint_exact, " \
                 "images_extracted.footprint_type FROM images INNER JOIN " \
                 "images_extracted ON images.image_id=images_extracted.image_id " \
                 "WHERE footprint_type ='calc' "
    data = ctd.get_data_from_db(sql_string, catch=False)

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):

        image_id = row["image_id"]

        if overwrite is False and os.path.isfile( fld + "/" + image_id + ".tif"):
            continue

        # prepare image
        image = liff.load_image_from_file(image_id)
        image = rb.remove_borders(image, image_id)
        image = ri.resize_image(image, (500, 500))

        # Sample data from row
        footprint = shapely.wkt.loads(row['footprint_exact'])
        azimuth = row["azimuth"]

        _ , _ , _, _, _ = gc.georef_calc(image_id, fld, image=image, footprint=footprint, azimuth=azimuth)


if __name__ == "__main__":
    create_low_res()