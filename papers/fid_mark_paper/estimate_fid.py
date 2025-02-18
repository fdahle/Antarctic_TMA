
flight_paths = ["1821", "1816", "1833", "2137", "1825", "2136",
                       "2143", "1826", "1813", "2141",
                "2140", "2073", "1822", "1827", "1684", "2142",
                       "1824", "1846", "2139", "2075"]

import src.base.connect_to_database as ctd
conn = ctd.establish_connection()

import src.estimate.estimate_fid_mark as efm

for fl in flight_paths:
    sql_string = ("SELECT * FROM images_fid_points WHERE "
                  f"substring(image_id, 3, 4)='{fl}'")
    data = ctd.execute_sql(sql_string, conn)
    print(data.shape)
    for key in [1,2,3,4,5,6,7,8]:

        # remove all rows where the fid_mark is already extracted
        kData = data[data[f'fid_mark_{key}_x'].isnull()]

        # iterate over the rows and print the image_id
        for i, row in kData.iterrows():

            image_id = row['image_id']

            coords = efm.estimate_fid_mark(image_id, str(key), conn=conn)

            if coords is None:
                continue

            # update the database
            sql_string = (f"UPDATE images_fid_points SET "
                          f"fid_mark_{key}_x={coords[0]},"
                          f"fid_mark_{key}_y={coords[1]},"
                          f"fid_mark_{key}_estimated=True, "
                          f"fid_mark_{key}_extraction_date=now() "
                          f"WHERE image_id='{image_id}'")
            ctd.execute_sql(sql_string, conn)
            print(row['image_id'], coords)

