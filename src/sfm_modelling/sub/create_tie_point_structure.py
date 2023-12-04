import glob
import os
import numpy as np

import base.connect_to_db as ctd
import base.print_v as p
import base.remove_borders as rb
import base.load_image_from_file as liff

import sfm_modelling.sub.resample_tie_points as rtp

r_digits = 2 # digits for rounding

debug_add_border = True
if debug_add_border:
    p.print_v("ctps: DEBUG ADD BORDER IS ACTIVE", color="red")

def create_tie_point_structure(project_folder, catch=True):

    # check if the project folder is really existing
    assert os.path.isdir(project_folder), f"'{project_folder}' is not a path to an existing folder"

    img_folder = project_folder + "/images_orig"

    # create homol if not existing
    homol_folder = project_folder + "/Homol"
    if not os.path.exists(homol_folder):
        os.makedirs(homol_folder)

    # get all image_ids from the images folder
    tif_files = glob.glob(os.path.join(img_folder, '*.tif')) + glob.glob(os.path.join(img_folder, '*.tiff'))
    ids = [os.path.splitext(os.path.basename(file))[0] for file in tif_files]

    # create a list string from the python list
    ids = "(" + str(ids)[1:-1] + ")"

    # get the tie-points from the database
    sql_string = "SELECT image_1_id, image_2_id, " \
                 "tie_points_image_1_filtered, tie_points_image_2_filtered, " \
                 "quality_filtered " \
                 "FROM images_tie_points WHERE " \
                 f"image_1_id in {ids} and image_2_id in {ids}"
    data = ctd.get_data_from_db(sql_string, catch=False)

    if data.shape[0] == 0:
        p.print_v("No images found to create a tie-point structure", color="red")
        exit()

    for index, row in data.iterrows():

        id_1 = row["image_1_id"]
        id_2 = row["image_2_id"]

        # get the points
        points_1_str = row["tie_points_image_1_filtered"]
        points_1 = np.asarray([list(map(float, point.split(','))) for point in points_1_str.split(";")])

        # add borders
        if debug_add_border:
            img_1 = liff.load_image_from_file(id_1)
            _, dims_1 = rb.remove_borders(img_1, id_1, return_edge_dims=True)

            points_1[:,0] = points_1[:,0] + dims_1[0]
            points_1[:,1] = points_1[:,1] + dims_1[2]

        # resample points for micmac
        points_1 = rtp.resample_tie_points(project_folder, id_1, points_1)

        if points_1 is None:
            if catch:
                continue
            else:
                raise ValueError

        points_2_str = row["tie_points_image_2_filtered"]
        points_2 = np.asarray([list(map(float, point.split(','))) for point in points_2_str.split(";")])

        # add borders
        if debug_add_border:
            img_2 = liff.load_image_from_file(id_2)
            _, dims_2 = rb.remove_borders(img_2, id_2, return_edge_dims=True)

            points_2[:,0] = points_2[:,0] + dims_2[0]
            points_2[:,1] = points_2[:,1] + dims_2[2]

        points_2 = rtp.resample_tie_points(project_folder, id_2, points_2)

        if points_2 is None:
            if catch:
                continue
            else:
                raise ValueError

        quality = row["quality_filtered"][1:-1].split(";")
        quality = list(filter(None, quality))
        quality = [float(i) for i in quality]
        quality = np.array(quality)

        # create the folders
        id_1_folder = homol_folder + "/PastisOIS-Reech_" + id_1 + ".tif"
        if not os.path.exists(id_1_folder):
            os.makedirs(id_1_folder)
        id_2_folder = homol_folder + "/PastisOIS-Reech_" + id_2 + ".tif"
        if not os.path.exists(id_2_folder):
            os.makedirs(id_2_folder)

        # get the path to the txt files
        id_1_path = id_2_folder + "/OIS-Reech_" + id_1 + ".tif.txt"
        id_2_path = id_1_folder + "/OIS-Reech_" + id_2 + ".tif.txt"

        print(id_1_path, id_2_path)

        # delete if exists
        if os.path.exists(id_1_path):
            os.remove(id_1_path)
        if os.path.exists(id_2_path):
            os.remove(id_2_path)

        with open(id_1_path, 'w') as f:
            for i in range(points_1.shape[0]):
                x_1 = round(points_1[i,0], r_digits)
                y_1 = round(points_1[i,1], r_digits)
                x_2 = round(points_2[i,0], r_digits)
                y_2 = round(points_2[i,1], r_digits)
                q = quality[i]

                f.write(f"{x_1} {y_1} {x_2} {y_2} {q}\n")

        with open(id_2_path, 'w') as f:
            for i in range(points_1.shape[0]):
                x_1 = round(points_2[i,0], r_digits)
                y_1 = round(points_2[i,1], r_digits)
                x_2 = round(points_1[i,0], r_digits)
                y_2 = round(points_1[i,1], r_digits)
                q = quality[i]

                f.write(f"{x_1} {y_1} {x_2} {y_2} {q}\n")


if __name__ == "__main__":

    fld = "/data_1/ATM/data_1/sfm/" \
          "projects/TEST"

    create_tie_point_structure(fld)