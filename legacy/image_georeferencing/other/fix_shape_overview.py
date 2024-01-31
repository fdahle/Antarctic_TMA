import base.connect_to_db as ctd
import base.modify_shape as ms
import base.print_v as p
import base.load_image_from_file as liff
import base.load_shape_data as lsd
import image_georeferencing.sub.calc_camera_position as ccp
import image_georeferencing.sub.convert_image_to_footprint as citf
import shapely.wkt

from shapely.geometry import Point, LineString
from rasterio import Affine


from tqdm import tqdm

debug_to_fix = ["satellite", "img", "calc"]

base_path = "/data_1/ATM/data_1/playground/georef4"

shape_paths = {
    "sat_footprints": f"{base_path}/overview/sat/sat_footprints.shp",
    "sat_photocenters": f"{base_path}/overview/sat/sat_photocenters.shp",
    "sat_error_vectors": f"{base_path}/overview/sat/sat_error_vectors.shp",
    "img_footprints": f"{base_path}/overview/img/img_footprints.shp",
    "img_photocenters": f"{base_path}/overview/img/img_photocenters.shp",
    "img_error_vectors": f"{base_path}/overview/img/img_error_vectors.shp",
    "calc_footprints": f"{base_path}/overview/calc/calc_footprints.shp",
    "calc_photocenters": f"{base_path}/overview/calc/calc_photocenters.shp",
    "calc_error_vectors": f"{base_path}/overview/calc/calc_error_vectors.shp",
}

overwrite = False

def fix_shape_overview():
    types = ["sat", "sat_est", "img", "calc"]
    methods = ["delete", "add"]

    olds_ids = []

    for type in types:
        sql_string = f"SELECT images_georef.image_id, status_{type}, method, " \
                     f"footprint_exact, position_exact, position_error_vector, footprint_type, " \
                     f"ST_AsText(position_approx) AS photocenter_approx, " \
                     f"t0, t1, t2, t3, t4, t5, t6, t7, t8 " \
                     f" FROM images_georef " \
                     f"INNER JOIN images_extracted ON images_georef.image_id = images_extracted.image_id " \
                     f"WHERE status_{type} = 'georeferenced' AND method='{type}'"
        data = ctd.get_data_from_db(sql_string, catch=False)

        # delete the entries not in the database
        if "delete" in methods:

            if type == "sat_est":
                _type = "sat"
            else:
                _type = type

            fp_data = lsd.load_shape_data(shape_paths[f"{_type}_footprints"])
            pc_data = lsd.load_shape_data(shape_paths[f"{_type}_photocenters"])
            ev_data = lsd.load_shape_data(shape_paths[f"{_type}_footprints"])

            ids_sql = data['image_id'].tolist()

            if type == "sat":
                old_ids = ids_sql
                continue
            elif type == "sat_est":
                ids_sql = ids_sql + old_ids

            fp_ids_shp = fp_data['image_id'].tolist()
            pc_ids_shp = pc_data['image_id'].tolist()
            ev_ids_shp = ev_data['image_id'].tolist()

            fp_differences = [item for item in fp_ids_shp if item not in ids_sql]
            pc_differences = [item for item in pc_ids_shp if item not in ids_sql]
            ev_differences = [item for item in ev_ids_shp if item not in ids_sql]

            for elem in fp_differences:
                ms.modify_shape(shape_paths[f"{_type}_footprints"],
                                elem, "delete")
            for elem in pc_differences:
                ms.modify_shape(shape_paths[f"{_type}_footprints"],
                                elem, "delete")
            for elem in ev_differences:
                ms.modify_shape(shape_paths[f"{_type}_footprints"],
                                elem, "delete")

        # add the correct entries
        if "add" in methods:
            counter = 0
            for index, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

                counter += 1

                _type = row['method']
                if _type == "sat":
                    continue

                if _type == "sat_est":
                    _type = "sat"

                img_id = row["image_id"]
                img_path = f"/data_1/ATM/data_1/playground/georef4/tiffs/{_type}"

                do_this_id = False

                if overwrite is True or ms.modify_shape(shape_paths[f"{_type}_footprints"], row['image_id'], "check") is False:
                    do_this_id = True
                if overwrite is True or ms.modify_shape(shape_paths[f"{_type}_photocenters"], row['image_id'], "check") is False:
                    do_this_id = True
                if overwrite is True or ms.modify_shape(shape_paths[f"{_type}_error_vectors"], row['image_id'], "check") is False:
                    do_this_id = True

                if do_this_id is False:
                    continue

                # load the geo-referenced image
                img = liff.load_image_from_file(img_id, img_path)

                # create transform object
                transform = Affine(row["t0"], row["t1"], row["t2"], row["t3"], row["t4"], row["t5"])

                # convert image to footprint
                exact_footprint = citf.convert_image_to_footprint(img, img_id, transform)

                if exact_footprint is None:
                    continue

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

                # check if already in
                if overwrite is True or ms.modify_shape(shape_paths[f"{_type}_footprints"], row['image_id'], "check") is False:
                    ms.modify_shape(shape_paths[f"{_type}_footprints"],
                                row['image_id'], "add", exact_footprint)
                if overwrite is True or ms.modify_shape(shape_paths[f"{_type}_photocenters"], row['image_id'], "check") is False:
                    ms.modify_shape(shape_paths[f"{_type}_photocenters"],
                                row['image_id'], "add", exact_position)
                if overwrite is True or ms.modify_shape(shape_paths[f"{_type}_error_vectors"], row['image_id'], "check") is False:
                    ms.modify_shape(shape_paths[f"{_type}_error_vectors"],
                                row['image_id'], "add", error_vector_calc)

if __name__ == "__main__":
    fix_shape_overview()