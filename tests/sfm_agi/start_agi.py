"""run an agisoft project with given images"""

# Python imports
import os

# Library imports
import numpy as np
import pandas as pd
from shapely import wkt

# suppress warnings
os.environ['KMP_WARNINGS'] = '0'

# Local imports
import src.sfm_agi.run_agi as ra  # noqa

import src.base.connect_to_database as ctd  # noqa
import src.georef.snippets.calc_azimuth as ca  # noqa
import src.load.load_military_focal_length as lmfl  # noqa
import src.sfm_agi.old.run_agi_relative as rar  # noqa
import src.sfm_agi.old.run_agi_absolute as raa  # noqa
import src.other.extract.extract_ids_by_area as eia  # noqa

# define the image ids
#image_ids = ["CA214732V0045", "CA214732V0046", "CA214732V0047", "CA214732V0048"]
image_ids = ['CA214831L0071', 'CA214831L0072', 'CA214831L0073', 'CA214831L0074', 'CA214832V0070', 'CA214832V0071', 'CA214832V0072', 'CA214832V0073', 'CA214832V0074', 'CA214931L0155', 'CA214931L0156', 'CA214931L0157', 'CA214931L0158', 'CA214931L0159', 'CA214932V0155', 'CA214932V0156', 'CA214932V0157', 'CA214932V0158', 'CA215031L0251', 'CA215031L0252', 'CA215031L0253', 'CA215031L0254', 'CA215032V0251', 'CA215032V0252', 'CA215032V0253', 'CA215131L0296', 'CA215131L0297', 'CA215131L0298', 'CA215131L0299', 'CA215132V0295', 'CA215132V0296', 'CA215132V0297', 'CA215132V0298', 'CA215132V0299', 'CA215132V0300', 'CA215133R0296', 'CA215133R0297', 'CA215133R0298', 'CA215133R0299', 'CA215733R0041']

bounds = None

# project settings
project_name = "new_test_9"
overwrite = False
resume = True

# accuracy settings (None means not using it)
camera_accuracy = (100, 100, 100)  # x, y, z in m
gcp_accuracy = (20, 20, 20)

# input settings
limit_images = 0  # 0 means no limit
use_positions = False  # if true, camera position will be given to agisoft
use_rotations = True  # if true, camera rotations will be given to agisoft
only_vertical = False

# define the path to the image folders
path_image_folder = "/data/ATM/data_1/aerial/TMA/downloaded"
georef_table = "images_extracted"

# check if we have at least 3 images
if len(image_ids) < 3:
    raise ValueError("Need at least 3 images")

# Convert the list to a string that looks like a tuple
image_ids_string = ', '.join(f"'{image_id}'" for image_id in image_ids)

# create conn to the database
conn = ctd.establish_connection()

# get all required data from the database
sql_string = "SELECT ie.image_id, ie.focal_length, ie.height, ie.altimeter_value, " \
             "i.altitude, ST_AsText(ie.footprint_exact) AS footprint_exact, " \
             "ST_AsText(ie.position_exact) AS position_exact, ie.azimuth_exact " \
             "FROM images_extracted ie JOIN images i ON ie.image_id=i.image_id " \
             f"WHERE ie.image_id in ({image_ids_string})"
data = ctd.execute_sql(sql_string, conn)

# order images by image_id
data = data.sort_values(by='image_id')

# give a warning if the number of images is not the same as the number of image_ids
if len(data) != len(image_ids):
    difference = set(image_ids) - set(data['image_id'])
    print(f"Warning: {len(difference)} images not found in the database")

if only_vertical:
    # filter for images where 'V' is in the image_id
    data = data[data['image_id'].str.contains('V')]

if limit_images > 1:
    data = data.head(limit_images)

# get the number of unique flight paths (2:6 of image_id)
flight_paths = set([image_id[2:6] for image_id in data['image_id']])

# convert the geometries to shapely objects
data['footprint_exact'] = data['footprint_exact'].apply(wkt.loads)
data['position_exact'] = data['position_exact'].apply(wkt.loads)

# print the number of images without positions
print(f"Number of images without positions: {data['position_exact'].isnull().sum()}/{len(data)}")
print(f"Number of images without footprints: {data['footprint_exact'].isnull().sum()}/{len(data)}")

# remove images without positions or footprints
print("Removing images without positions or footprints..")
data = data.dropna(subset=['position_exact', 'footprint_exact'])

# get the focal length from the military calibration
data['focal_length'] = data['image_id'].apply(lambda x: lmfl.load_military_focal_length(x, None, conn))

# print the number of images without focal length
print(f"Number of images without focal length: {data['focal_length'].isnull().sum()}/{len(data)}")

# check if there are any missing focal lengths
if data['focal_length'].isnull().sum() > 0:
    print("Using default value for focal length..")
    data['focal_length'] = data['focal_length'].fillna(154.43)


# define function to extract z
def _get_z(z_row):
    feet_to_meters = 0.3048  # Conversion factor from feet to meters
    if pd.notnull(z_row['height']):
        return z_row['height'] * feet_to_meters
    elif pd.notnull(z_row['altimeter_value']):
        return z_row['altimeter_value'] * feet_to_meters
    elif pd.notnull(z_row['altitude']) and z_row['altitude'] != -99999:
        return z_row['altitude'] * feet_to_meters

    # default value
    return 22000 * feet_to_meters


# create x, y, z columns
data['pos_x'] = data['position_exact'].apply(lambda x: x.x)
data['pos_y'] = data['position_exact'].apply(lambda x: x.y)
data['pos_z'] = data.apply(_get_z, axis=1)
data['pos_tuple'] = data.apply(lambda _row: (_row['pos_x'], _row['pos_y'], _row['pos_z']), axis=1)

# create the different dicts from the dataframe
focal_length_dict = data.set_index('image_id')['focal_length'].to_dict()
footprint_dict = data.set_index('image_id')['footprint_exact'].to_dict()
position_dict = data.set_index('image_id')['pos_tuple'].to_dict()

# create accuracy dict
if camera_accuracy is not None:
    accuracy_dict = {image_id: camera_accuracy for image_id in data['image_id']}
else:
    accuracy_dict = None

# create rotation dict
rotation_dict = {}
for image_id in data['image_id']:

    # get the row for the image
    row = data[data['image_id'] == image_id]

    # yaw is the exact azimuth
    yaw = row['azimuth_exact'].values[0]

    # account for the different coordinate system
    yaw = 360 - yaw + 90
    yaw = round(yaw, 2)
    yaw = yaw % 360

    # pitch is always 0
    pitch = 0

    # set roll depending on image direction
    if "V" in image_id:
        roll = 0
    elif "L" in image_id:
        roll = 30
    elif "R" in image_id:
        roll = 360 - 30
    else:
        raise ValueError(f"Unknown image direction in {image_id}")

    rotation_dict[image_id] = (yaw, pitch, roll)

if len(flight_paths) > 1:
    print(f"Azimuth cannot be used due to {len(flight_paths)} different flight paths")
    azimuth = None
else:
    # get constant values
    azimuth = np.mean(data['azimuth_exact'])
    azimuth = 360 - azimuth + 90

if use_positions is False:
    position_dict = None

if use_rotations is False:
    rotation_dict = None

print("TEEEMP")
accuracy_dict = None

# create lst with absolute paths
image_ids = data['image_id'].tolist()
images_paths = [os.path.join(path_image_folder, image + ".tif") for image in image_ids]


ra.run_agi(project_name, images_paths,
           focal_lengths=focal_length_dict, camera_footprints=footprint_dict,
           camera_positions=position_dict, camera_rotations=rotation_dict,
           camera_accuracies=accuracy_dict, gcp_accuracy=gcp_accuracy,
           azimuth=azimuth, absolute_bounds=bounds,
           overwrite=overwrite, resume=resume)
