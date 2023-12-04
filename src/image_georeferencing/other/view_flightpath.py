import pandas as pd
from shapely.wkt import loads

import base.connect_to_db as ctd

import display.display_shapes as ds

catch=False

def view_flightpath(flight_path):

    # get all images with this flight path
    sql_string = "SELECT image_id, SUBSTRING(image_id, 3, 4) AS flight_path, " \
                 "ST_AsText(footprint_exact) AS footprint_exact, " \
                 "ST_AsText(footprint_approx) as footprint_approx, " \
                 "ST_AsText(position_exact) AS position_exact, " \
                 "ST_AsText(position_approx) AS position_approx, " \
                 "position_error_vector, footprint_Type FROM images_extracted " \
                 f"WHERE SUBSTRING(image_id, 3, 4) ='{flight_path}' AND " \
                 f"image_id LIKE '%V%'"
    data = ctd.get_data_from_db(sql_string, catch=catch)

    sql_string_georef = "SELECT * FROM images_georef " \
                        f"WHERE SUBSTRING(image_id, 3, 4) ='{flight_path}'"
    data_georef = ctd.get_data_from_db(sql_string_georef)
    data = pd.merge(data, data_georef, how='right', on='image_id')

    # init empty lists
    ids_sat, footprints_sat, points_sat = [], [], []
    ids_sat_est, footprints_sat_est, points_sat_est = [], [], []
    ids_img, footprints_img, points_img = [], [], []
    ids_calc, footprints_calc, points_calc = [], [], []

    for idx, row in data.iterrows():
        if row['method'] == "sat" and row["status_sat"] == "georeferenced":
            ids_sat.append(row['image_id'][-4:])
            footprints_sat.append(loads(row['footprint_exact']))
            points_sat.append(loads(row['position_exact']))

        if row['method'] == "sat_est" and row["status_sat_est"] == "georeferenced":
            ids_sat_est.append(row['image_id'][-4:])
            footprints_sat_est.append(loads(row['footprint_exact']))
            points_sat_est.append(loads(row['position_exact']))

        if row['method'] == "img" and row["status_img"] == "georeferenced":
            ids_img.append(row['image_id'][-4:])
            footprints_img.append(loads(row['footprint_exact']))
            points_img.append(loads(row['position_exact']))

        if row['method'] == "calc" and row["status_calc"] == "georeferenced":
            ids_calc.append(row['image_id'][-4:])
            footprints_calc.append(loads(row['footprint_exact']))
            points_calc.append(loads(row['position_exact']))

    nr_georef = len(ids_sat) + len(ids_sat_est) + len(ids_img) + len(ids_calc)

    if nr_georef == 0:
        return

    ds.display_shapes([footprints_sat, points_sat,
                       footprints_sat_est, points_sat_est,
                       footprints_img, points_img,
                       footprints_calc, points_calc],
                      subtitles=[None, ids_sat,
                                 None, ids_sat_est,
                                 None, ids_img,
                                 None, ids_calc],
                      colors=["green", "black", "green", "black", "lightgreen", "black", "yellow", "black"],
                      title=flight_path + " " + str(len(ids_sat)) + "," + str(len(ids_sat_est)) + "," + str(len(ids_img)) + "," + str(len(ids_calc))
                      )

if __name__ == "__main__":

    _sql_string = "SELECT DISTINCT SUBSTRING(image_id, 3, 4) AS flight_path FROM images_extracted"
    _data = ctd.get_data_from_db(_sql_string)

    for _i, _row in _data.iterrows():

        view_flightpath(_row['flight_path'])
