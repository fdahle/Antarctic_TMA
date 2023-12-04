import shapely.wkt

from rasterio import Affine
from shapely.geometry import Point, LineString
from tqdm import tqdm

import base.connect_to_db as ctd
import base.load_image_from_file as liff

import image_georeferencing.sub.calc_camera_position as ccp
import image_georeferencing.sub.convert_image_to_footprint as citf

types = ["sat", "sat_est", "img", "calc"]

overwrite = True

def save_footprints_to_db():

    for type in types:
        sql_string = f"SELECT images_georef.image_id, status_{type}, method, " \
                     f"footprint_exact, position_exact, position_error_vector, footprint_type, " \
                     f"ST_AsText(position_approx) AS photocenter_approx, " \
                     f"t0, t1, t2, t3, t4, t5, t6, t7, t8 " \
                     f" FROM images_georef " \
                     f"INNER JOIN images_extracted ON images_georef.image_id = images_extracted.image_id " \
                     f"WHERE status_{type} = 'georeferenced' AND method='{type}'"
        data = ctd.get_data_from_db(sql_string, catch=False)


        for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

            if overwrite is False and row['footprint_type'] != None:
                continue

            method = row['method']
            if method == "sat_est":
                method = "sat"

            img_id = row["image_id"]
            img_path = f"/data_1/ATM/data_1/playground/georef4/tiffs/{method}"

            # load the geo-referenced image
            img, transform_img = liff.load_image_from_file(img_id, img_path, return_transform=True)

            if img is None:
                print(f"No image for {img_id}")
                continue

            # create transform object
            transform = Affine(row["t0"],row["t1"],row["t2"],row["t3"],row["t4"],row["t5"])

            # convert image to footprint
            exact_footprint = citf.convert_image_to_footprint(img, img_id, transform)

            # get approx data
            photocenter_approx = shapely.wkt.loads(row['photocenter_approx'])

            if photocenter_approx is None:
                print(f"No photocenter for {img_id}")
                continue

            approx_x, approx_y = photocenter_approx.x, photocenter_approx.y

            # calculate exact x and y of photocenter
            exact_x, exact_y = ccp.calc_camera_position(exact_footprint)

            # create a shapely point that we can save in a db
            exact_position = Point(exact_x, exact_y)

            # create error vector
            error_vector_calc = LineString([(exact_x, exact_y), (approx_x, approx_y)])

            dx = exact_x - approx_x
            dy = exact_y - approx_y

            # update database
            sql_string = f"UPDATE images_extracted SET " \
                         f"footprint_exact=ST_GeomFromText('{exact_footprint.wkt}'), " \
                         f"position_exact=ST_GeomFromText('{exact_position.wkt}'), " \
                         f"position_error_vector='{dx};{dy}', " \
                         f"footprint_type='{row['method']}' " \
                         f"WHERE image_id='{img_id}'"
            ctd.edit_data_in_db(sql_string, catch=False, verbose=False)

if __name__ == "__main__":
    save_footprints_to_db()
