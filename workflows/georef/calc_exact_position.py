from shapely import wkt, LineString
from tqdm import tqdm

import src.base.connect_to_database as ctd
import src.georef.snippets.calc_camera_position as ccp

table = "images_georef"
overwrite = True
flight_paths = [2163]

def calc_exact_position():
    nr_updated = 0
    # create connection to database
    conn = ctd.establish_connection()

    # get the data from the database
    sql_string = ("SELECT image_id, "
                  "st_astext(footprint_exact) AS footprint_exact, "
                  "st_astext(position_exact) AS position_exact "
                  "FROM images_georef "
                  "WHERE footprint_exact IS NOT NULL")

    data = ctd.execute_sql(sql_string, conn)

    print(data.shape)

    if flight_paths is not None:
        str_fp = [str(fp) for fp in flight_paths]
        data = data[data['image_id'].str[2:6].isin(str_fp)]

    print(data.shape)

    # iterate over all rows
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

        # check if we should overwrite
        if overwrite is False and row['position_exact'] is not None:
            continue

        image_id = row['image_id']

        # convert the geometries to shapely objects
        footprint_exact = wkt.loads(row['footprint_exact'])

        # calc the position
        position_exact = ccp.calc_camera_position(image_id, footprint_exact)

        # update the database
        sql_string = (f"UPDATE {table} "
                      f"SET position_exact = ST_GeomFromText('{position_exact}', 3031) "
                      f"WHERE image_id = '{image_id}'")
        ctd.execute_sql(sql_string, conn)

        nr_updated = nr_updated + 1

    print(f"Updated {nr_updated} entries")

if __name__ == "__main__":
    calc_exact_position()