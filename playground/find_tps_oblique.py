import copy

import src.base.connect_to_database as ctd
import src.base.create_mask as cm
import src.base.find_tie_points as ftp
import src.load.load_image as li
import src.other.extract.extract_ids_by_area as eiba
import src.display.display_images as di

# connect to database
conn = ctd.establish_connection()

# set image id
id_img = "CA184832V0150"
buffer = 1000 # in m

# get footprint for that id
sql_string = "SELECT ST_AsText(footprint_exact) as footprint " \
             f"FROM images_extracted " \
             f"WHERE image_id = '{id_img}'"
data = ctd.execute_sql(sql_string, conn)
footprint = data['footprint'].iloc[0]

# get the intersecting images
intersect_ids = eiba.extract_ids_by_area(aoi=footprint)

# add image_id to the list
sky_ids = copy.deepcopy(intersect_ids)
sky_ids.append(id_img)

# get the correct rotation for the intersecting ids
sql_string = "SELECT image_id, sky_is_correct FROM images WHERE image_id IN ('" + "', '".join(sky_ids) + "')"
sky_data = ctd.execute_sql(sql_string, conn)

# init tie-point detector
tpd = ftp.TiePointDetector('lightglue', min_conf=0.8)

# Load the image
img = li.load_image(id_img)
mask = cm.create_mask(img, use_default_fiducials=True,
                      default_fid_position=500,
                      use_database=True, image_id=id_img)

# check for image_id if sky is correct
if sky_data[sky_data['image_id'] == id_img]['sky_is_correct'].iloc[0] is False:

    # flip the image 180 degree
    img = img[::-1, ::-1]
    mask = mask[::-1, ::-1]

# shuffle intersect_ids
import random
random.shuffle(intersect_ids)

print(intersect_ids)

for other_id in intersect_ids:
    other_img = li.load_image(other_id)
    other_mask = cm.create_mask(other_img, use_default_fiducials=True,
                                use_database=False, image_id=other_id)

    if "V" in id_img and "V" in other_id:
        print(f"Both images ({id_img} and {other_id}) are vertical")
        continue

    if sky_data[sky_data['image_id'] == other_id]['sky_is_correct'].iloc[0] is False:
        other_img = other_img[::-1, ::-1]
        other_mask = other_mask[::-1, ::-1]

    tps, conf = tpd.find_tie_points(img, other_img, mask1=mask, mask2=other_mask)

    print(f"Found {tps.shape[0]} tie-points for {id_img} and {other_id}")

    if tps.shape[0] < 5:
        continue

    style_config = {
        'title':f'{id_img} {other_id}'
    }
    di.display_images([img, other_img],
                      overlays=[mask, other_mask],
                      tie_points=tps, tie_points_conf=conf,
                      style_config=style_config)

    #di.display_images([img, other_img, mask, other_mask])

