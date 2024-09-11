# Python imports
import os

# Library imports
import numpy as np
import pandas as pd
from shapely import wkt

# suppress warnings
os.environ['KMP_WARNINGS'] = '0'

# Local imports
import src.sfm_agi.run_agi as ra

import src.base.connect_to_database as ctd  # noqa
import src.georef.snippets.calc_azimuth as ca  # noqa
import src.sfm_agi.old.run_agi_relative as rar  # noqa
import src.sfm_agi.old.run_agi_absolute as raa  # noqa
import src.other.extract.extract_ids_by_area as eia  # noqa

"""
# define the image ids
image_ids = [
    "CA181332V0124", "CA181332V0125", "CA181332V0127", "CA181332V0128",
    "CA181332V0129", "CA181332V0130", "CA181332V0131", "CA181332V0132",
    "CA181332V0133", "CA184332V0068", "CA184332V0069", "CA184332V0071",
    "CA184332V0072", "CA184332V0073", "CA184332V0074", "CA184332V0075",
    "CA184332V0077", "CA184332V0078", "CA184332V0079"
]
"""
# some settings
project_name = "tp_gcp_test"
use_positions = False  # if true, camera position and rotations will be given to agisoft
horizontal_camera_accuracy = 100  # in m
vertical_camera_accuracy = 100  # in m
gcp_accuracy = "TODO"
limit_images = 4
only_vertical = True
overwrite = False
resume = True

min_x = -2358616.2502435083
min_y = 1225594.5941433292
max_x = -2399188.311136363
max_y = 1293414.1377820491
bounds = [min_x, min_y, max_x, max_y]

# get the image ids
image_ids = eia.extract_ids_by_area(bounds, footprint_type="exact",
                                    check_clusters=True)

# define the path to the image folder
path_image_folder = "/data/ATM/data_1/aerial/TMA/downloaded"
georef_table = "images_extracted"

# create lst with absolute paths
images = [os.path.join(path_image_folder, image + ".tif") for image in image_ids]

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
images = [images[image_ids.index(image_id)] for image_id in data['image_id']]

# add images as a column to data
data['image_path'] = images

# remove images without a footprint
data = data.dropna(subset=['footprint_exact'])

if only_vertical:
    # filter for images where 'V' is in the image_id
    data = data[data['image_id'].str.contains('V')]

if limit_images is not None:
    data = data.head(limit_images)

# get the number of unique flight paths (2:6 of image_id)
flight_paths = set([image_id[2:6] for image_id in data['image_id']])

# get images back as list
images = data['image_path'].tolist()

# convert the geometries to shapely objects
data['footprint_exact'] = data['footprint_exact'].apply(wkt.loads)
data['position_exact'] = data['position_exact'].apply(wkt.loads)


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


print("WARNING: Using default value for focal length.")
data['focal_length'] = data['focal_length'].fillna(154.43)

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
accuracy_dict = {}
for image_id in data['image_id']:
    accuracy_dict[image_id] = (horizontal_camera_accuracy, horizontal_camera_accuracy, vertical_camera_accuracy)

# create rotation dict
rotation_dict = {}
for image_id in data['image_id']:

    # get the row for the image
    row = data[data['image_id'] == image_id]

    # yaw is the exact azimuth
    yaw = row['azimuth_exact'].values[0]

    # account for the different coordinate system
    yaw = 360 - yaw + 90

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
    rotation_dict = None
    accuracy_dict = None

print("TEEMP")
focal_length_dict = None

ra.run_agi(project_name, images,
           focal_lengths=focal_length_dict, camera_footprints=footprint_dict,
           camera_positions=position_dict, camera_rotations=rotation_dict,
           camera_accuracies=accuracy_dict,
           azimuth=azimuth, absolute_bounds=bounds, overwrite=overwrite, resume=resume)
