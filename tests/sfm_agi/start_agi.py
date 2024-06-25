# Library imports
import os
os.environ['KMP_WARNINGS'] = '0'
import pandas as pd
from shapely import wkt

# Local imports
import src.base.connect_to_database as ctd

# define the image ids
#image_ids = ["CA180132V0094", "CA180132V0095", "CA180132V0096", "CA180132V0097"]
             #"CA180132V0098", "CA180132V0099", "CA180132V0100", "CA180132V00101"]

# image_ids = ['CA184932V0241', 'CA184932V0242', 'CA184932V0243', 'CA184932V0244']

image_ids = ['CA184832V0146', 'CA184832V0147', 'CA184832V0148', 'CA184832V0149', 'CA184832V0150']

# get the first six characters of the first image id
project_name = image_ids[0][:6]

project_name = project_name

# get only the first 3 images
#image_ids = image_ids[:3]

# define the path to the image folder
path_image_folder = "/data_1/ATM/data_1/aerial/TMA/downloaded"
georef_table = "images_georef_3"
#georef_table = "images_extracted"

# create lst with absolute paths
images = [os.path.join(path_image_folder, image + ".tif") for image in image_ids]

# create sql list string for image_ids
image_ids_string = "','".join(image_ids)
image_ids_string = "('" + image_ids_string + "')"

# create conn to the database
conn = ctd.establish_connection()

# create a dict with the focal lengths
sql_string = f"SELECT image_id, focal_length " \
             f"FROM images_extracted WHERE image_id in {image_ids_string}"
focal_length_data = ctd.execute_sql(sql_string, conn)
focal_length_dict = focal_length_data.set_index('image_id')['focal_length'].to_dict()

sql_string = f"SELECT images_extracted.image_id, images_extracted.height, " \
             f"images_extracted.altimeter_value, images.altitude, " \
             f"ST_AsText({georef_table}.position_exact) AS position_exact " \
             f"FROM images_extracted JOIN images ON images_extracted.image_id=images.image_id " \
             f"WHERE images_extracted.image_id in {image_ids_string}"

if georef_table != "images_extracted":
    sql_string = sql_string.replace("WHERE",
                       f"JOIN {georef_table} ON images_extracted.image_id={georef_table}.image_id WHERE")
position_data = ctd.execute_sql(sql_string, conn)

position_data['position_geometry'] = position_data['position_exact'].apply(wkt.loads)
position_data['x'] = position_data['position_geometry'].apply(lambda geom: geom.x if geom is not None else None)
position_data['y'] = position_data['position_geometry'].apply(lambda geom: geom.y if geom is not None else None)

# Define a function to choose the correct altitude value
def get_z(row):
    feet_to_meters = 0.3048  # Conversion factor from feet to meters
    if pd.notnull(row['height']):
        return row['height'] * feet_to_meters
    elif pd.notnull(row['altimeter_value']):
        return row['altimeter_value'] * feet_to_meters
    elif pd.notnull(row['altitude']) and row['altitude'] != -99999:
        return row['altitude'] * feet_to_meters
    return 22000 * feet_to_meters

# Apply the function to get z values
position_data['z'] = position_data.apply(get_z, axis=1)

position_data = position_data[['image_id', 'x', 'y', 'z']]

# create a rotation dict
import src.georef.snippets.calc_azimuth as ca
rotation_dict = {}
for image_id in image_ids:
    azimuth = ca.calc_azimuth(image_id, conn)
    rotation_dict[image_id] = (azimuth, 0, 0)  # yaw, pitch, roll


# Local imports
import src.sfm_agi.init_agi as ia

# init the agisoft project
ia.init_agi(project_name, images, focal_lengths=focal_length_dict,
            camera_positions=position_data, camera_rotations=rotation_dict)
