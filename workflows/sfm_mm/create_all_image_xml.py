# Library imports
import os
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd
import src.sfm_mm.prepare.create_image_xml as cix

# Variables
output_folder = "/data/ATM/data_1/sfm/xml/images/"
overwrite = False


def create_all_image_xml():

    # create connection to the database
    conn = ctd.establish_connection()

    # get all images
    sql_string = "SELECT * FROM images"
    data = ctd.execute_sql(sql_string, conn)

    # get all fiducial marks from the database
    sql_string = "SELECT image_id, " \
                 "fid_mark_1_x, fid_mark_1_y, fid_mark_2_x, fid_mark_2_y, " \
                 "fid_mark_3_x, fid_mark_3_y, fid_mark_4_x, fid_mark_4_y, " \
                 "fid_mark_5_x, fid_mark_5_y, fid_mark_6_x, fid_mark_6_y, " \
                 "fid_mark_7_x, fid_mark_7_y, fid_mark_8_x, fid_mark_8_y " \
                 "FROM images_fid_points"
    fid_data = ctd.execute_sql(sql_string, conn)

    # save how many entries are updated
    updated_entries = 0

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Create image xml for {image_id} "
                             f"({updated_entries} already updated)")

        # get the fiducial marks
        fid_marks = fid_data[fid_data['image_id'] == image_id]

        # check if there is nan in any of the fiducial marks
        if fid_marks.isnull().values.any():
            continue

        # define the output path
        output_path = os.path.join(output_folder, f"MeasuresIm-{image_id}.tif.xml")

        # check if the file already exists
        if os.path.exists(output_path) and not overwrite:
            continue

        # create the xml file
        cix.create_image_xml(image_id, fid_marks, output_path)

        # update the file path in the database
        sql_string = f"UPDATE images_file_paths SET path_xml_file='{output_path}' " \
                     f"WHERE image_id='{image_id}'"

        ctd.execute_sql(sql_string, conn, add_timestamp=False)

        # save how many entries are updated
        updated_entries = updated_entries + 1


if __name__ == "__main__":
    create_all_image_xml()