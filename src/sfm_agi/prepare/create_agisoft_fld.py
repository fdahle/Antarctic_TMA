# Package imports
import cv2
import os
import shutil

# Custom imports
import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.load.load_image as li

# Constants
PATH_IMAGE_FLD = "/data_1/ATM/data_1/aerial/TMA/downloaded"
PATH_AGISOFT_FLD = "/home/fdahle/SFTP/staff-umbrella/ATM/agisoft"


def create_agisoft_fld(image_ids, project_name=None):
    # check if the base folder exists
    if not os.path.exists(PATH_AGISOFT_FLD):
        raise FileNotFoundError(f"Folder {PATH_AGISOFT_FLD} does not exist (Is the server mounted?)")

    # connect to database
    conn = ctd.establish_connection()

    # set project name to tma if not given
    if project_name is None:
        project_name = image_ids[0][2:6]

    # create the project folder
    project_folder = os.path.join(PATH_AGISOFT_FLD, project_name)
    if not os.path.exists(project_folder):
        os.mkdir(project_folder)

    # create sub folder for images
    image_folder = os.path.join(project_folder, "images")
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    # copy the images to the folder
    for image_id in image_ids:

        # skip images that are already in the folder
        if os.path.exists(os.path.join(image_folder, f"{image_id}.tif")):
            continue

        shutil.copy(os.path.join(PATH_IMAGE_FLD, f"{image_id}.tif"),
                    os.path.join(image_folder, f"{image_id}.tif"))

    # create mask folder
    if not os.path.exists(os.path.join(project_folder, "masks")):
        os.mkdir(os.path.join(project_folder, "masks"))

    sql_string = "SELECT * from images_fid_points WHERE image_id IN ('" + "', '".join(image_ids) + "')"
    data_fid_marks = ctd.execute_sql(sql_string, conn)

    sql_string = "SELECT * from images_extracted WHERE image_id IN ('" + "', '".join(image_ids) + "')"
    data_extracted = ctd.execute_sql(sql_string, conn)

    # create masks
    for image_id in image_ids:

        if os.path.exists(os.path.join(project_folder, "masks", f"{image_id}_mask.tif")):
            continue

        image = li.load_image(image_id)

        # Get the fid marks for the specific image_id
        fid_marks_row = data_fid_marks.loc[data_fid_marks['image_id'] == image_id].squeeze()

        # Create fid mark dict using dictionary comprehension
        fid_dict = {i: (fid_marks_row[f'fid_mark_{i}_x'], fid_marks_row[f'fid_mark_{i}_y']) for
                    i in range(1, 5)}

        # get the text boxes of the image
        text_string = data_extracted.loc[data_extracted['image_id'] == image_id]['text_bbox'].iloc[0]

        # make all text strings to lists
        if len(text_string) > 0 and "[" not in text_string:
            text_string = "[" + text_string + "]"

        # create text-boxes list
        text_boxes = [tuple(group) for group in eval(text_string.replace(";", ","))]

        # load the mask
        mask = cm.create_mask(image, fid_dict, text_boxes, use_default_fiducials=True)

        # adapt mask for agisoft
        mask = mask.astype('uint8') * 255

        # save the mask
        cv2.imwrite(os.path.join(project_folder, "masks", f"{image_id}_mask.tif"), mask)

    # get some information about the images
    sql_string = "SELECT image_id, " \
                 "ST_AsText(position_exact) AS position_exact, focal_length, height " \
                 "FROM images_extracted " \
                 "WHERE image_id IN ('" + "', '".join(image_ids) + "')"
    data = ctd.execute_sql(sql_string, conn)

    # convert height from feet to meters
    data['height'] = data['height'] * 0.3048

    # save focal length
    data[["image_id", "focal_length"]].to_csv(os.path.join(project_folder, "focal_length.csv"), index=False)

    # get x and y coordinates
    data['x'] = data['position_exact'].str.split(" ").str[0].str[6:].astype(float)
    data['y'] = data['position_exact'].str.split(" ").str[1].str[:-1].astype(float)

    # Append '.tif' to every entry in the 'image_id' column
    data['image_id'] = data['image_id'].apply(lambda x: f"{x}.tif")

    # save camera positions
    data[["image_id", "x", "y", "height"]].to_csv(os.path.join(project_folder, "cameras.csv"), index=False)
