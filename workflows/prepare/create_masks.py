# Package imports
from tqdm import tqdm

# Custom imports
import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.load.load_image as li


def create_masks():
    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images and their masks from the database
    sql_string = "SELECT image_id, path_mask FROM images"
    data = ctd.execute_sql(sql_string, conn)

    # get all fid marks from the database
    sql_string_fid_marks = "Select_image_id, " \
                           "fid_mark_1_x, fid_mark_1_y, " \
                           "fid_mark_2_x, fid_mark_2_y, " \
                           "fid_mark_3_x, fid_mark_3_y, " \
                           "fid_mark_4_x, fid_mark_4_y " \
                           "FROM images_fid_points"
    data_fid_marks = ctd.execute_sql(sql_string_fid_marks, conn)

    # get position of text boxes
    sql_string_boxes = "SELECT image_id, text_bbox FROM images_extracted"
    data_boxes = ctd.execute_sql(sql_string_boxes, conn)

    # join all dataframes by the image_id
    data = data.merge(data_fid_marks, on='image_id', how='left')
    data = data.merge(data_boxes, on='image_id', how='left')

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        # load the image
        image = li.load_image(image_id)

        # get the fid marks as a dict
        fid_marks = {
            '1': (row['fid_mark_1_x'], row['fid_mark_1_y']),
            '2': (row['fid_mark_2_x'], row['fid_mark_2_y']),
            '3': (row['fid_mark_3_x'], row['fid_mark_3_y']),
            '4': (row['fid_mark_4_x'], row['fid_mark_4_y']),
        }

        # get the text boxes as a list
        text_boxes = row['text_bbox']
        text_boxes = text_boxes.split(";") if text_boxes is not None else []

        # create mask for the image
        mask = cm.create_mask(image, fid_marks, text_boxes)

        # last check if the mask has identical shape to the image
        if mask.shape != image.shape:
            if catch:
                continue
            else:
                raise RuntimeError(f"Mask shape {mask.shape} does not match image shape {image.shape}")


if __name__ == "__main__":
    create_masks()
