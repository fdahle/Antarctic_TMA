import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.export.export_tiff as et
import src.load.load_image as li
import src.sfm_agi.snippets.create_adapted_mask as cam


_image_ids = ['CA184832V0146', 'CA184832V0147', 'CA184832V0148', 'CA184832V0149', 'CA184832V0150']
_save_fld = "/home/fdahle/Desktop/agi_test/masks"

def create_agi_masks(image_ids, save_fld):

    conn = ctd.establish_connection()

    # iterate all images
    for image_id in image_ids:

        image = li.load_image(image_id)

        # get fid marks
        sql_string_fid_marks = f"SELECT * FROM images_fid_points WHERE image_id='{image_id}'"
        data_fid_marks = ctd.execute_sql(sql_string_fid_marks, conn)

        # Get the fid marks for the specific image_id
        fid_marks_row = data_fid_marks.loc[data_fid_marks['image_id'] == image_id].squeeze()

        # Create fid mark dict using dictionary comprehension
        fid_dict = {str(i): (fid_marks_row[f'fid_mark_{i}_x'], fid_marks_row[f'fid_mark_{i}_y']) for i in range(1, 5)}

        # get the text information
        sql_string_text = f"SELECT * FROM images_extracted WHERE image_id='{image_id}'"
        data_text = ctd.execute_sql(sql_string_text, conn)

        # get the text boxes of the image
        text_string = data_text.loc[data_text['image_id'] == image_id]['text_bbox'].iloc[0]

        if len(text_string) > 0 and "[" not in text_string:
            text_string = "[" + text_string + "]"

        # create text-boxes list
        text_boxes = [list(group) for group in eval(text_string.replace(";", ","))]

        # create base mask
        base_mask = cm.create_mask(image, fid_dict, text_boxes)

        # adapt for agisoft
        agi_mask = cam.create_adapted_mask(base_mask, image_id, conn)

        # save the mask
        save_path = f"{save_fld}/{image_id}_mask.png"

        et.export_tiff(agi_mask, save_path, overwrite=True)

if __name__ == '__main__':
    create_agi_masks(_image_ids, _save_fld)