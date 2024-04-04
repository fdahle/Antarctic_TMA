import src.base.connect_to_database as ctd
import src.load.load_image as li
import src.prepare.extract_text as et

from tqdm import tqdm

overwrite = False
retry = False


def extract_text_from_images():

    # establish connection to psql
    conn = ctd.establish_connection()

    # get all images with their text from the database
    sql_string = "SELECT image_id, text_bbox, text_content FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # save how many entries are updated
    updated_entries = 0

    # filter out the images that already have text
    if overwrite is False:
        data = data[data['text_content'].isnull()]

    # filter out the images that have been tried before
    if retry is False:
        data = data[data['text_content'] != '']

    # loop over all images
    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']

        pbar.set_postfix_str(f"Extract text for {image_id} "
                             f"({updated_entries} already updated)")

        # load the image
        try:
            image = li.load_image(image_id)

            # extract text from the image
            text, text_pos, _ = et.extract_text(image, catch=False)
        except (Exception,):
            text, text_pos = None, None

        if text is None:
            text, text_pos = "", ""
        else:
            # remove all ";" from the text as we use it as a separator
            text = [t.replace(";", "") for t in text]

            # also remove "'" as it is used in the sql string
            text = [t.replace("'", "") for t in text]

            text = ";".join(text)
            text_pos = ";".join(["(" + ",".join([str(int(x)) for x in t]) + ")" for t in text_pos])

        # update the database
        sql_string = f"UPDATE images_extracted SET " \
                     f"text_bbox='{text_pos}', " \
                     f"text_content='{text}' WHERE image_id='{image_id}'"
        ctd.execute_sql(sql_string, conn)
        updated_entries += 1


if __name__ == '__main__':
    extract_text_from_images()
