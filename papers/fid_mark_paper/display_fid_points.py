
flight_paths = ["1822"]
direction = "V"

import src.base.connect_to_database as ctd
conn = ctd.establish_connection()

import src.other.extract.extract_ids_by_flightpath as eibf

image_ids = eibf.extract_ids_by_area(flight_paths, conn=conn)

for image_id in image_ids:
    # connect to database
    sql_string = "SELECT * FROM images_fid_points WHERE image_id='{}'".format(image_id)
    data = ctd.execute_sql(sql_string, conn)

    if direction not in image_id:
        continue

    # get x, y coordinates of fid marks
    fid_marks = []
    for i in range(5, 9):
        if data[f"fid_mark_{i}_x"].isna().values[0]:
            continue
        fid_marks.append((data[f"fid_mark_{i}_x"].values[0], data[f"fid_mark_{i}_y"].values[0]))

    import src.load.load_image as li
    img = li.load_image(image_id)

    if len(fid_marks) == 0:
        continue

    print(image_id, fid_marks)

    import src.display.display_images as di
    di.display_images(img, points=[fid_marks])