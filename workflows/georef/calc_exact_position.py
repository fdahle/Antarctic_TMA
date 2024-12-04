from shapely import wkt, LineString
from tqdm import tqdm

import src.base.connect_to_database as ctd
import src.georef.snippets.calc_camera_position as ccp

overwrite=False

def calc_exact_position():

    # create connection to database
    conn = ctd.establish_connection()

    # get the data from the database
    sql_string = ("SELECT image_id, st_astext(footprint_exact) AS footprint_exact "
                  "FROM images_georef "
                  "WHERE footprint_exact IS NOT NULL")
    if overwrite is False:
        sql_string += " AND position_exact IS NULL"

    data = ctd.execute_sql(sql_string, conn)

    # iterate over all rows
    for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

        # convert the geometries to shapely objects
        footprint_exact = wkt.loads(row['footprint_exact'])

        # calc the position
        position_exact = ccp.calc_camera_position(footprint_exact)

        # update the database
        sql_string = (f"UPDATE {table} "
                      f"SET error_vector = ST_GeomFromText('{error_vector}', 3031) "
                      f"WHERE image_id = '{row['image_id']}'")
        ctd.execute_sql(sql_string, conn)

if __name__ == "__main__":
    calc_exact_position()