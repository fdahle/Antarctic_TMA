import glob
import os

from tqdm import tqdm

import src.load.load_image as li
import src.load.load_shape_data as lsd
import src.load.load_transform as lt

import src.georef.snippets.verify_image_geometry as vig
import src.georef.snippets.verify_image_position as vip

georef_folder = "/data_1/ATM/data_1/georef"
georef_type = "sat"

DISTANCE_THRESHOLD = 100


def verify_footprints(path_georef_fld, check_geometry=True, check_position=True):
    # get all tifs in the folder
    pattern = os.path.join(path_georef_fld + "/" + georef_type, '*.tif')
    tif_files = glob.glob(pattern)

    invalid_images = []

    # load the shape file
    if check_position:
        shape_data = lsd.load_shape_data(georef_folder + "/" + georef_type + ".shp")
    else:
        shape_data = None

    # iterate all geo-referenced images
    for file in (pbar := tqdm(tif_files)):

        image_id = os.path.basename(file)[:-4]

        pbar.set_postfix_str(f"Check {image_id}")

        if check_geometry:
            transform_path = georef_folder + "/" + georef_type + "/" + image_id + "_transform.txt"

            image = li.load_image(image_id)
            transform = lt.load_transform(transform_path)

            valid_geometry, _ = vig.verify_image_geometry(image, transform)
        else:
            valid_geometry = True

        if check_position:

            # get geometry of image_id from shape_data
            try:
                image_data = shape_data[shape_data['image_id'] == image_id]
                image_geom = image_data['geometry'].iloc[0]

                # get geometries of same flight number
                filtered_data = shape_data[shape_data['image_id'].str[2:6] == image_id[2:6]]
                filtered_data = filtered_data[filtered_data['image_id'] != image_id]
                flight_geoms = filtered_data['geometry'].values.tolist()

                if len(flight_geoms) < 3:
                    valid_position = True
                else:
                    valid_position = vip.verify_image_position(image_geom, flight_geoms, DISTANCE_THRESHOLD)
                    print(valid_position)
            except:
                valid_position = False
        else:
            valid_position = True

        if valid_position is False or valid_geometry is False:
            invalid_images.append(image_id)

    print(f"{len(invalid_images)} of {len(tif_files)} are invalid.")
    print(invalid_images)


if __name__ == "__main__":
    verify_georef(georef_folder)
