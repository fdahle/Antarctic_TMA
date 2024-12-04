from tqdm import tqdm
from shapely import wkt

import src.base.connect_to_database as ctd
import src.display.display_shapes as ds
import src.georef.snippets.calc_camera_position as ccp

overwrite = False
image_ids = None

# Debug variables
debug_display_position = True

def calc_approximate_positions():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get images and their parameters from the database
    sql_string = ("SELECT image_id, "
                  "st_astext(footprint_approx) AS footprint_approx, "
                  "st_astext(position_approx) AS position_approx "
                  "FROM images_extracted")

    # add conditionally image ids
    if image_ids is not None:
        if len(image_ids) == 1:
            sql_string += f" WHERE image_id = '{image_ids[0]}'"
        else:
            sql_string += f" WHERE image_id IN {tuple(image_ids)}"

    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(data.shape)

    # filter out the images that already have a footprint
    if overwrite is False:
        data = data[data['position_approx'].isnull()]

    print(data.shape[0], "images to process")
    exit()
    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):
        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Calc approx footprint for {image_id} "
                             f"({updated_entries} already updated)")

        # get footprint
        footprint_txt = row['footprint_approx']
        footprint = wkt.loads(footprint_txt)

        # get center as well
        center = ccp.calc_camera_position(image_id, footprint)

        if debug_display_position:
            ds.display_shapes([footprint, center])
        continue

        sql_string = f"UPDATE images_extracted SET " \
                     f"position_approx=ST_GeomFromText('{center}') " \
                     f"WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)

        updated_entries += 1


if __name__ == "__main__":
    calc_approximate_positions()
