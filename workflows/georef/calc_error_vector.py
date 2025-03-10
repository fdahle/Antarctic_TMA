from shapely import wkt, LineString
from tqdm import tqdm

import src.base.connect_to_database as ctd

table = "images_georef"
overwrite=False
flight_paths = None

def calc_error_vector():

    nr_updated = 0

    # create connection to database
    conn = ctd.establish_connection()

    # get the data from the database
    sql_string = (f"SELECT {table}.image_id, "
                  f"ST_astext({table}.position_exact) AS position_exact, "
                  f"ST_astext(st_transform(st_setsrid(images.point, 4326), 3031)) AS point, "
                  f"ST_astext({table}.error_vector) AS error_vector "
                  f"FROM {table} JOIN images ON {table}.image_id = images.image_id")
    data = ctd.execute_sql(sql_string, conn)

    print(data.shape)

    if flight_paths is not None:
        str_fp = [str(fp) for fp in flight_paths]
        data = data[data['image_id'].str[2:6].isin(str_fp)]

    print(data.shape)

    # iterate over all rows
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

        # check if we should overwrite
        if overwrite is False and row['error_vector'] is not None:
            continue

        # check if we have all the data
        if row['position_exact'] is None or row['point'] is None:
            continue

        # convert the geometries to shapely objects
        exact_point = wkt.loads(row['position_exact'])
        approx_point = wkt.loads(row['point'])

        # get a vector from the two points
        error_vector = LineString([exact_point, approx_point])

        if row['image_id'] in ["CA182332V0059", "CA182332V0060", "CA182332V0061", "CA182332V0062", "CA182332V0063"]:
            print(row['image_id'])
            print(error_vector)
            print(exact_point)
            # get length of the vector
            print(error_vector.length)

        # update the database
        sql_string = (f"UPDATE {table} "
                      f"SET error_vector = ST_GeomFromText('{error_vector}', 3031) "
                      f"WHERE image_id = '{row['image_id']}'")
        ctd.execute_sql(sql_string, conn)

        nr_updated = nr_updated + 1

    print(f"Updated {nr_updated} entries")


if __name__ == "__main__":
    calc_error_vector()