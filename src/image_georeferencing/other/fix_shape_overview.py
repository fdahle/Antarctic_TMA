import base.connect_to_db as ctd
import base.modify_shape as ms
import base.print_v as p

from tqdm import tqdm

debug_to_fix = ["satellite", "img", "calc"]

base_path = "/data_1/ATM/data_1/playground/georef4"

shape_paths = {
    "satellite_footprints": f"{base_path}/overview/sat/sat_footprints.shp",
    "satellite_photocenters": f"{base_path}/overview/sat/sat_photocenters.shp",
    "satellite_error_vectors": f"{base_path}/overview/sat/sat_error_vectors.shp",
    "img_footprints": f"{base_path}/overview/img/img_footprints.shp",
    "img_photocenters": f"{base_path}/overview/img/img_photocenters.shp",
    "img_error_vectors": f"{base_path}/overview/img/img_error_vectors.shp",
    "calc_footprints": f"{base_path}/overview/calc/calc_footprints.shp",
    "calc_photocenters": f"{base_path}/overview/calc/calc_photocenters.shp",
    "calc_error_vectors": f"{base_path}/overview/calc/calc_error_vectors.shp",
}

def fix_shape_overview():

    sql_string = "SELECT image_id, " \
                 "ST_AsText(footprint_exact) AS footprint_exact, " \
                 "ST_AsText(position_exact) AS position_exact," \
                 "position_error_vector, footprint_type " \
                 "FROM images_extracted WHERE footprint_type IS NOT NULL"
    data = ctd.get_data_from_db(sql_string, catch=False)

    counter = 0

    for index, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        p.print_v(f"{counter} ids updated", verbose=True, pbar=pbar)

        # check if we want to fix it
        if row['footprint_type'] not in debug_to_fix:
            continue

        status_fp = ms.modify_shape(shape_paths[f"{row['footprint_type']}_footprints"],
                        row['image_id'], "check")
        status_pc = ms.modify_shape(shape_paths[f"{row['footprint_type']}_photocenters"],
                        row['image_id'], "check")

        if status_pc is False or status_fp is False:
            counter += 1

        if status_fp is False:
            ms.modify_shape(shape_paths[f"{row['footprint_type']}_footprints"],
                            row['image_id'], "add", row['footprint_exact'])
        if status_pc is False:
            ms.modify_shape(shape_paths[f"{row['footprint_type']}_photocenters"],
                        row['image_id'], "add", row['position_exact'])

if __name__ == "__main__":
    fix_shape_overview()