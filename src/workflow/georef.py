from tqdm import tqdm

# import base function
import src.base.connect_to_database as ctd
import src.base.create_mask as cm

# import load functions
import src.load.load_image as li

# import georef functions
import src.georef.georef_sat as gs
import src.georef.georef_img as gi
import src.georef.georef_calc as gc

# import georef snippet functions
import src.georef.snippets.apply_transform as at
import src.georef.snippets.verify_image as vi

def georef():

    # init the geo-reference objects
    georefSat = gs.GeorefSatellite()
    georefImg = gi.GeorefImage()
    georefCalc = gc.GeorefCalc()

    # establish connection to the database
    conn = ctd.establish_connection()

    # get month and angles from the database
    sql_string_images = "SELECT image_id, azimuth, date_month FROM images"
    data_images = ctd.execute_sql(sql_string_images, conn)

    # get footprints
    sql_string_footprints = "SELECT image_id, ST_AsText(footprint_approx) AS footprint_approx FROM images_extracted"
    data_footprints = ctd.execute_sql(sql_string_footprints, conn)

    # geo-reference with satellite
    for image_id in tqdm(input_ids):

        # load the image
        image = li.load_image(image_id)

        # load the mask
        mask = cm.create_mask(image, fid_marks, text_boxes)

        # get the approx footprint of the image
        approx_footprint = data_footprints.loc[data_footprints['image_id'] == image_id]

        # get the azimuth of the image
        azimuth = data_images.loc[data_images['image_id'] == image_id]['angle']

        transform, residuals, tps, conf = georefSat.georeference(image, approx_footprint, mask, angle, month)

    # geo-reference with image
    for image_id in tqdm(input_ids):


def _handle_georef_results():

    # verify the image
    vi.verify_image(image, transform)

    # save the geo-referenced image

    # save the transform

    # save the points


if __name__ == "__main__":

    input_ids = ""
