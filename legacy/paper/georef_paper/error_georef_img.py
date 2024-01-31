import shapely
import numpy as np
import rasterio

import base.connect_to_db as ctd
import base.load_image_from_file as liff
import base.remove_borders as rb
import base.rotate_image as ri

import display.display_images as di
import display.display_tiepoints as dt

import image_georeferencing.sub.apply_gcps as ag
import image_georeferencing.sub.filter_tp_outliers as fto

import  image_tie_points.find_tie_points as ftp

import display.display_shapes as ds

id = "CA184632V0333"
direction = "forward"  # can be back or forward
max_img = 30
ref_fld = "/data_1/ATM/data_1/playground/georef4/tiffs/sat/"
output_fld = "/data_1/ATM/data_1/playground/georef4/for_paper/image_error/"

# load azimuth
sql_string = f"SELECT azimuth FROM images WHERE image_id='{id}'"
data = ctd.get_data_from_db(sql_string)
azimuth = data['azimuth'].iloc[0]

# load footprint
sql_string = f"SELECT ST_AsText(footprint_exact) AS footprint_exact" \
             f" FROM images_extracted WHERE image_id='{id}'"
data = ctd.get_data_from_db(sql_string)
footprint_wkt = data['footprint_exact'].iloc[0]
footprint = shapely.from_wkt(footprint_wkt)



# get the image nr
short_id = id[:-4]
image_nr = int(id[-4:])

if direction == "back":
    all_ids = list(range(image_nr, image_nr + max_img))
elif direction == "forward":
    all_ids = list(range(image_nr, image_nr - max_img, -1))

print(all_ids)

for nr in all_ids[:-1]:
    if direction == "back":
        next_nr = nr + 1
    elif direction == "forward":
        next_nr = nr - 1

    id = short_id + f"{str(nr):0>4}"
    next_id = short_id + f"{str(next_nr):0>4}"

    print(id, next_id)

    # load images
    img, transform = liff.load_image_from_file(id, image_path=ref_fld, return_transform=True, catch=False)

    next_img = liff.load_image_from_file(next_id)
    next_img = rb.remove_borders(next_img, next_id)
    next_img = ri.rotate_image(next_img, azimuth)

    # find tps
    tps, conf = ftp.find_tie_points(img, next_img)

    tps, conf = fto.filter_tp_outliers(tps, conf,
                                                 use_ransac=True,
                                                 ransac_threshold=10,
                                                 use_angle=True,
                                                 angle_threshold=5)

    #dt.display_tiepoints([img, next_img], points=tps)

    # make some tps absolute
    #tps[:, :2] = np.array(
    #    rasterio.transform.xy(transform, tps[:, 0], tps[:, 1])).transpose()

    tps[:, 0] = tps[:, 0] * transform[0]
    tps[:, 1] = tps[:, 1] * transform[4]
    tps[:, 0] = tps[:, 0] + transform[2]
    tps[:, 1] = tps[:, 1] + transform[5]

    print("x_ref", np.amin(tps[:,0]), np.amax(tps[:,0]))
    print("y_ref", np.amin(tps[:,1]), np.amax(tps[:,1]))
    print("x_unref", np.amin(tps[:,2]), np.amax(tps[:,2]))
    print("y_unref", np.amin(tps[:,3]), np.amax(tps[:,3]))

    new_transform, res = ag.apply_gcps(output_fld + next_id + ".tif", next_img, tps, "rasterio", return_error=True)

    ref_fld = output_fld

#ds.display_shapes([footprint])
