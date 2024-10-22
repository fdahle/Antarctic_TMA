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
image_ids = [
    "CA181332V0124", "CA181332V0125", "CA181332V0127", "CA181332V0128",
    "CA181332V0129", "CA181332V0130", "CA181332V0131", "CA181332V0132",
    "CA181332V0133", "CA184332V0068", "CA184332V0069", "CA184332V0071",
    "CA184332V0072", "CA184332V0073", "CA184332V0074", "CA184332V0075",
    "CA184332V0077", "CA184332V0078", "CA184332V0079"
]

#min_x = -2412960
#min_y = 1246047
#max_x = -2393107
#max_y = 1261346
#bounds = [min_x, min_y, max_x, max_y]

# get the image ids
#image_ids = eia.extract_ids_by_area(bounds, footprint_type="exact",
#                                    check_clusters=True)

image_ids = ["CA213731L0072", "CA213731L0073", "CA213731L0074", "CA213731L0075", "CA213731L0076", "CA213731L0077", "CA213731L0078", "CA213731L0079", "CA213731L0080",
"CA213731L0081", "CA213731L0082", "CA213731L0083", "CA213731L0084", "CA213731L0085", "CA213731L0086", "CA213731L0087", "CA213731L0088", "CA213731L0089",
"CA213731L0090", "CA213732V0072", "CA213732V0073", "CA213732V0074", "CA213732V0075", "CA213732V0076", "CA213732V0077", "CA213732V0078", "CA213732V0079",
"CA213732V0080", "CA213732V0081", "CA213732V0082", "CA213732V0083", "CA213732V0084", "CA213732V0085", "CA213732V0086", "CA213732V0087", "CA213732V0088",
"CA213732V0089", "CA213732V0090", "CA213733R0072", "CA213733R0073", "CA213733R0074", "CA213733R0075", "CA213733R0076", "CA213733R0077", "CA213733R0078",
"CA213733R0079", "CA213733R0080", "CA213733R0081", "CA213733R0082", "CA213733R0083", "CA213733R0084", "CA213733R0085", "CA213733R0086", "CA213733R0087",
"CA213733R0088", "CA213733R0089", "CA213733R0090", "CA214731L0029", "CA214731L0030", "CA214731L0031", "CA214731L0032", "CA214731L0033", "CA214731L0034",
"CA214731L0035", "CA214731L0036", "CA214731L0037", "CA214731L0038", "CA214731L0039", "CA214731L0040", "CA214731L0041", "CA214731L0042", "CA214731L0043",
"CA214731L0044", "CA214731L0045", "CA214731L0046", "CA214731L0047", "CA214731L0048", "CA214731L0049", "CA214732V0029", "CA214732V0030", "CA214732V0031",
"CA214732V0032", "CA214732V0033", "CA214732V0034", "CA214732V0035", "CA214732V0036", "CA214732V0037", "CA214732V0038", "CA214732V0039", "CA214732V0040",
"CA214732V0041", "CA214732V0042", "CA214732V0043", "CA214732V0044", "CA214732V0045", "CA214732V0046", "CA214732V0047", "CA214732V0048", "CA214732V0049",
"CA214733R0029", "CA214733R0030", "CA214733R0031", "CA214733R0032", "CA214733R0033", "CA214733R0034", "CA214733R0035", "CA214733R0036", "CA214733R0037",
"CA214733R0038", "CA214733R0039", "CA214733R0040", "CA214733R0041", "CA214733R0042", "CA214733R0043", "CA214733R0044", "CA214733R0045", "CA214733R0046",
"CA214733R0047", "CA214733R0048", "CA214733R0049", "CA214831L0069", "CA214831L0070", "CA214831L0071", "CA214831L0072", "CA214831L0073", "CA214831L0074",
"CA214831L0075", "CA214831L0076", "CA214831L0077", "CA214831L0078", "CA214831L0079", "CA214831L0080", "CA214831L0081", "CA214831L0082", "CA214831L0083",
"CA214831L0084", "CA214831L0085", "CA214832V0069", "CA214832V0070", "CA214832V0071", "CA214832V0072", "CA214832V0073", "CA214832V0074", "CA214832V0075",
"CA214832V0076", "CA214832V0077", "CA214832V0078", "CA214832V0079", "CA214832V0080", "CA214832V0081", "CA214832V0082", "CA214832V0083", "CA214832V0084",
"CA214832V0085", "CA214833R0069", "CA214833R0070", "CA214833R0071", "CA214833R0072", "CA214833R0073", "CA214833R0074", "CA214833R0075", "CA214833R0076",
"CA214833R0077", "CA214833R0078", "CA214833R0079", "CA214833R0080", "CA214833R0081", "CA214833R0082", "CA214833R0083", "CA214833R0084", "CA214833R0085",
"CA214931L0137", "CA214931L0138", "CA214931L0139", "CA214931L0140", "CA214931L0141", "CA214931L0142", "CA214931L0143", "CA214931L0144", "CA214931L0145",
"CA214931L0146", "CA214931L0147", "CA214931L0148", "CA214931L0149", "CA214931L0150", "CA214931L0151", "CA214931L0152", "CA214931L0153", "CA214931L0154",
"CA214931L0155", "CA214931L0156", "CA214931L0157", "CA214932V0137", "CA214932V0138", "CA214932V0139", "CA214932V0140", "CA214932V0141", "CA214932V0142",
"CA214932V0143", "CA214932V0144", "CA214932V0145", "CA214932V0146", "CA214932V0147", "CA214932V0148", "CA214932V0149", "CA214932V0150", "CA214932V0151",
"CA214932V0152", "CA214932V0153", "CA214932V0154", "CA214932V0155", "CA214932V0156", "CA214932V0157", "CA214933R0137", "CA214933R0138", "CA214933R0139",
"CA214933R0140", "CA214933R0141", "CA214933R0142", "CA214933R0143", "CA214933R0144", "CA214933R0145", "CA214933R0146", "CA214933R0147", "CA214933R0148",
"CA214933R0149", "CA214933R0150", "CA214933R0151", "CA214933R0152", "CA214933R0153", "CA214933R0154", "CA214933R0155", "CA214933R0156", "CA214933R0157",
"CA215031L0250", "CA215031L0251", "CA215031L0252", "CA215031L0253", "CA215031L0254", "CA215031L0255", "CA215031L0256", "CA215031L0257", "CA215031L0258",
"CA215031L0259", "CA215031L0260", "CA215031L0261", "CA215031L0262", "CA215031L0263", "CA215031L0264", "CA215031L0265", "CA215031L0266", "CA215032V0250",
"CA215032V0251", "CA215032V0252", "CA215032V0253", "CA215032V0254", "CA215032V0255", "CA215032V0256", "CA215032V0257", "CA215032V0258", "CA215032V0259",
"CA215032V0260", "CA215032V0261", "CA215032V0262", "CA215032V0263", "CA215032V0264", "CA215032V0265", "CA215032V0266", "CA215033R0250", "CA215033R0251",
"CA215033R0252", "CA215033R0253", "CA215033R0254", "CA215033R0255", "CA215033R0256", "CA215033R0257", "CA215033R0258", "CA215033R0259", "CA215033R0260",
"CA215033R0261", "CA215033R0262", "CA215033R0263", "CA215033R0264", "CA215033R0265", "CA215033R0266", "CA215131L0283", "CA215131L0284", "CA215131L0285",
"CA215131L0286", "CA215131L0287", "CA215131L0288", "CA215131L0289", "CA215131L0290", "CA215131L0291", "CA215131L0292", "CA215131L0293", "CA215131L0294",
"CA215131L0295", "CA215131L0296", "CA215131L0297", "CA215131L0298", "CA215131L0299", "CA215131L0300", "CA215131L0301", "CA215131L0302", "CA215132V0283",
"CA215132V0284", "CA215132V0285", "CA215132V0286", "CA215132V0287", "CA215132V0288", "CA215132V0289", "CA215132V0290", "CA215132V0291", "CA215132V0292",
"CA215132V0293", "CA215132V0294", "CA215132V0295", "CA215132V0296", "CA215132V0297", "CA215132V0298", "CA215132V0299", "CA215132V0300", "CA215132V0301",
"CA215132V0302", "CA215133R0283", "CA215133R0284", "CA215133R0285", "CA215133R0286", "CA215133R0287", "CA215133R0288", "CA215133R0289", "CA215133R0290",
"CA215133R0291", "CA215133R0292", "CA215133R0293", "CA215133R0294", "CA215133R0295", "CA215133R0296", "CA215133R0297", "CA215133R0298", "CA215133R0299",
"CA215133R0300", "CA215133R0301", "CA215133R0302", "CA215231L0340", "CA215231L0341", "CA215231L0342", "CA215231L0343", "CA215231L0344", "CA215231L0345",
"CA215231L0346", "CA215231L0347", "CA215231L0348", "CA215231L0349", "CA215231L0350", "CA215231L0351", "CA215231L0352", "CA215231L0353", "CA215231L0354",
"CA215331L0407", "CA215331L0408", "CA215331L0409", "CA215331L0410", "CA215331L0411", "CA215331L0412", "CA215331L0413", "CA215331L0414", "CA215331L0415",
"CA215331L0416", "CA215331L0417", "CA215331L0418", "CA215331L0419", "CA215331L0420", "CA215331L0421", "CA215331L0422", "CA215331L0423", "CA215331L0424",
"CA215331L0425", "CA215331L0426", "CA215331L0427", "CA215331L0428", "CA215331L0429", "CA215331L0430", "CA215331L0431", "CA215331L0432", "CA215331L0433",
"CA215331L0434", "CA215331L0435", "CA215331L0436", "CA215332V0407", "CA215332V0408", "CA215332V0409", "CA215332V0410", "CA215332V0411", "CA215332V0412",
"CA215332V0413", "CA215332V0414", "CA215332V0415", "CA215332V0416", "CA215332V0417", "CA215332V0418", "CA215332V0419", "CA215332V0420", "CA215332V0421",
"CA215332V0422", "CA215332V0423", "CA215332V0424", "CA215332V0425", "CA215332V0426", "CA215332V0427", "CA215332V0428", "CA215332V0429", "CA215332V0430",
"CA215332V0431", "CA215332V0432", "CA215332V0433", "CA215332V0434", "CA215332V0435", "CA215332V0436", "CA215333R0413", "CA215333R0414", "CA215333R0415",
"CA215333R0416", "CA215333R0417", "CA215333R0418", "CA215333R0419", "CA215333R0420", "CA215333R0421", "CA215333R0422", "CA215333R0423", "CA215333R0424",
"CA215333R0425", "CA215333R0426", "CA215333R0427", "CA215333R0428", "CA215333R0429", "CA215333R0430", "CA215333R0431", "CA215333R0432", "CA215333R0433",
"CA215333R0434", "CA215333R0435", "CA215333R0436", "CA215333R0437", "CA215333R0438", "CA215731L0040", "CA215731L0041", "CA215731L0042", "CA215731L0043",
"CA215731L0044", "CA215731L0045", "CA215731L0046", "CA215731L0047", "CA215731L0048", "CA215731L0049", "CA215731L0050", "CA215731L0051", "CA215731L0052",
"CA215731L0053", "CA215731L0054", "CA215731L0055", "CA215731L0056", "CA215731L0057", "CA215731L0058", "CA215732V0040", "CA215732V0041", "CA215732V0042",
"CA215732V0043", "CA215732V0044", "CA215732V0045", "CA215732V0046", "CA215732V0047", "CA215732V0048", "CA215732V0049", "CA215732V0050", "CA215732V0051",
"CA215732V0052", "CA215732V0053", "CA215732V0054", "CA215732V0055", "CA215732V0056", "CA215732V0057", "CA215732V0058", "CA215733R0040", "CA215733R0041",
"CA215733R0042", "CA215733R0043", "CA215733R0044", "CA215733R0045", "CA215733R0046", "CA215733R0047", "CA215733R0048", "CA215733R0049", "CA215733R0050",
"CA215733R0051", "CA215733R0052", "CA215733R0053", "CA215733R0054", "CA215733R0055", "CA215733R0056", "CA215733R0057", "CA215733R0058"]
bounds = None
print("Used Image ids:")
print(image_ids)

# project settings
project_name = "agi_ryan_vertical"
overwrite = False
resume = True

# accuracy settings (None means not using it)
camera_accuracy = (100, 100, 100)  # x, y, z in m
gcp_accuracy = (20, 20, 20)


# input settings
limit_images = 0  # 0 means no limit
use_positions = False  # if true, camera position will be given to agisoft
use_rotations = True  # if true, camera rotations will be given to agisoft
only_vertical = True

# define the path to the image folder
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
