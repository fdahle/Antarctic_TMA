import numpy as np

import src.base.connect_to_database as ctd
from tqdm import tqdm

def update_image_azimuth(overwrite=False):

    conn = ctd.establish_connection()

    sql_string = "SELECT image_id, CONCAT(SUBSTRING(image_id, 3, 4), SUBSTRING(image_id, 10, 4)) AS triplet, azimuth_exact FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # get all different image triplets from pandas dataframe
    triplets = data['triplet'].unique()

    # keep track of updated azimuths
    nr_updated = 0

    for triplet in (pbar := tqdm(triplets,total=len(triplets))):

        pbar.set_postfix_str(f"Updated {nr_updated} azimuths")

        # get all images from the same triplet
        triplet_images = data.loc[data['triplet'] == triplet]

        # check if the triplet has 3 images
        if triplet_images.shape[0] != 3:
            continue

        # get the left and right ids
        left_id = triplet_images.loc[triplet_images['image_id'].str.contains('L')]['image_id'].values[0]
        right_id = triplet_images.loc[triplet_images['image_id'].str.contains('R')]['image_id'].values[0]

        # get the azimuths of the triplets
        vertical_azimuth = triplet_images.loc[triplet_images['image_id'].str.contains('V')]['azimuth_exact'].values[0]
        left_azimuth = triplet_images.loc[triplet_images['image_id'].str.contains('L')]['azimuth_exact'].values[0]
        right_azimuth = triplet_images.loc[triplet_images['image_id'].str.contains('R')]['azimuth_exact'].values[0]

        # check if the azimuth of the vertical image is not null
        if vertical_azimuth is None or np.isnan(vertical_azimuth):
            continue

        # calculate the azimuth for the other two images
        new_azimuth_left = round((vertical_azimuth - 90) % 360, 2)
        new_azimuth_right = round((vertical_azimuth + 90) % 360, 2)

        if left_azimuth is None or np.isnan(left_azimuth) or overwrite:
            sql_string = f"UPDATE images_extracted SET azimuth_exact={new_azimuth_left} WHERE image_id='{left_id}'"
            ctd.execute_sql(sql_string, conn)
            nr_updated += 1
        if right_azimuth is None or np.isnan(right_azimuth) or overwrite:
            sql_string = f"UPDATE images_extracted SET azimuth_exact={new_azimuth_right} WHERE image_id='{right_id}'"
            ctd.execute_sql(sql_string, conn)
            nr_updated += 1

if __name__ == "__main__":
    update_image_azimuth()