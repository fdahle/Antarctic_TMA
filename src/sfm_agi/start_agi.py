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
import src.other.extract.extract_ids_by_area as eia  # noqa

glacier_name = None

"""
# define the image ids
image_ids = ["CA214732V0045", "CA214732V0046", "CA214732V0047", "CA214732V0048"]
image_ids = ['CA214831L0071', 'CA214831L0072', 'CA214831L0073', 'CA214831L0074', 'CA214832V0070', 'CA214832V0071', 'CA214832V0072', 'CA214832V0073', 'CA214832V0074', 'CA214931L0155', 'CA214931L0156', 'CA214931L0157', 'CA214931L0158', 'CA214931L0159', 'CA214932V0155', 'CA214932V0156', 'CA214932V0157', 'CA214932V0158', 'CA215031L0251', 'CA215031L0252', 'CA215031L0253', 'CA215031L0254', 'CA215032V0251', 'CA215032V0252', 'CA215032V0253', 'CA215131L0296', 'CA215131L0297', 'CA215131L0298', 'CA215131L0299', 'CA215132V0295', 'CA215132V0296', 'CA215132V0297', 'CA215132V0298', 'CA215132V0299', 'CA215132V0300', 'CA215133R0296', 'CA215133R0297', 'CA215133R0298', 'CA215133R0299', 'CA215733R0041']
image_ids = ["CA216632V0280", "CA216632V0281", "CA216632V0282", "CA216632V0283",
             "CA216632V0284", "CA216632V0285", "CA216632V0286", "CA216632V0287"]
"""
image_ids = ["CA214732V0029", "CA214732V0030", "CA214732V0031", "CA214732V0032",
             "CA214732V0033", "CA214732V0034", "CA214732V0035", "CA214732V0036",
             "CA214732V0037"]
bounds = None



"""
project_source = "crane"
path_project_csv = f"/data/ATM/data_1/sfm/agi_data/research_areas/{project_source}.csv"
# read the csv
df = pd.read_csv(path_project_csv)
# get image_ids as list
image_ids = df["'image_id'"].tolist()
# remove all ' from every string
image_ids = [image_id.replace("'", "") for image_id in image_ids]
"""

glac_names = ["Aagaard Glacier", "Ahlmann Glacier", "Aitkenhead Glacier", "Ajax Icefall", "Alexander Island",
         "ALPHA GLACIER", "Anderson Glacier", "Andersson Island Ice Cap", "Andrew Glacier", "Anthony Glacier",
         "Aphrodite Glacier", "Arago Glacier", "Archer Glacier", "Arena Glacier", "Argentina Cove Glacier",
         "Athene Glacier", "Attlee Glacier", "Avsyuk Glacier", "Bader Glacier", "Bagshawe Glacier", "Balch Glacier",
         "Ball Glacier", "Baranowski Glacier", "Barnes Glacier", "Bayly Glacier", "Beaglehole Glacier",
         "Bellingshausen Dome", "Bertrand Ice Piedmont", "Beta Glacier", "Bevin Glacier", "Bilgeri Glacier",
         "Birley Glacier", "Blanchard Glacier", "Bleriot Glacier", "Bolton Glacier", "Boyana Glacier",
         "Boydell Glacier", "Breguet Glacier", "Breitfuss Glacier", "Broad Valley Glacier", "Bucher Glacier",
         "Bussey Glacier", "Cadman Glacier", "Cape Alexander Ice Cap a", "Cape Alexander Ice Cap b", "Cardell Glacier",
         "Carlson Glacier", "Case Island IC,", "Casey Glacier", "Cayley Glacier", "Centurian Glacier",
         "Chamberlin Glacier", "Channel Glacier", "Charcot Island IC,", "Charity Glacier",
         "Churchill Peninsula Ice Cap a", "Churchill Peninsula Ice Cap b", "Clarke Glacier", "Cole Peninsula Ice Cap",
         "Coley Glacier", "Comrie Glacier", "Contell Glacier", "Cordini Glacier", "Crane Glacier", "Cronus Glacier",
         "Crystal Glacier", "Daspit Glacier", "DAVIS DOME", "De Atley Island IC,", "Dead Glacier", "Dee Ice Piedmont",
         "Demorest Glacier", "Depot Glacier", "Deville Glacier", "Diplock Glacier", "Dobrowolski Glacier A",
         "Dobrowolski Glacier B", "Dobrudzha Glacier", "Domeyko Glacier", "Dorsey Island IC,", "Doyle Glacier",
         "Dragon Glacier", "Drummond Glacier", "Drygalski Glacier", "Ecology Glacier", "Eden Glacier",
         "Edgeworth Glacier", "Eldred Glacier", "Emerald Icefalls A", "Emerald Icefalls B", "Emerald Icefalls C",
         "Erskine Glacier", "Eureka Glacier", "Evans Glacier", "Field Glacier", "Flask Glacier", "Flint Glacier",
         "Forbes Glacier", "Forel Glacier", "Fourcade Glacier", "Franca Glacier", "Fricker Glacier",
         "Friederichsen Glacier", "Fuchs Ice Piedmont", "Funk Glacier", "Getman Ice Piedmont",
         "Glaciar Bahia Pastorizo", "Glaciar Bravo", "Glaciar Fuerza AÃÂ©rea", "Glaciar Nueve de Julio",
         "Glaciar Quito", "Glaciar Reichert", "Glaciar Solis", "Glacier de ViÃÂ©ville", "Goetel Glacier",
         "Goodwin Glacier", "Gould Glacier", "Gourdon Glacier", "Gran Glaciar Traub", "Gregory Glacier",
         "Harbour Glacier", "Hariot Glacier", "Hawkes Glacier", "Hearst Island IC,", "Heim Glacier", "Hektor Icefall A",
         "Hektor Icefall B", "Hektoria Glacier", "Henryk Glacier", "Hess Glacier", "Hippocrates Glacier",
         "Hobbs Glacier", "Hoek Glacier", "Hooper Glacier", "Hopkins Glacier", "Horatio Glacier", "Howarth Glacier",
         "Hugi Glacier", "Huidobro Ice Field", "Huntress Glacier", "Hurd (Sally/Argentina/Las Palmas) Glacier",
         "Hurley Glacier", "Huron Glacier", "Iliad Glacier", "Ineson Glacier", "Ineson Glacier,", "Johnsons Glacier",
         "Jones IS", "Jorum Glacier", "Kaliakra Glacier", "Koch Glacier", "Krak Glacier", "Krakowiak Glacier",
         "Krebs Glacier", "Ladies Icefall", "Laennec Glacier", "Lange Glacier", "Las Palmas Cove Glacier",
         "Latady Island IC,", "Lawrie Glacier", "Leay Glacier", "Leonardo Glacier", "Leppard Glacier", "Lever Glacier",
         "Lewis Glacier", "Lind Glacier", "Lister Glacier", "Lliboutry Glacier", "Luke Glacier", "Lurabee Glacier",
         "Macy Glacier", "Magura Glacier", "Maitland Glacier", "Malpighi Glacier", "Mamut Glacier", "Mapple Glacier",
         "Martin Glacier", "Matejko Icefall", "Matthes Glacier", "McCance Glacier", "McMorrin Glacier",
         "McNeile Glacier", "Melville Glacier", "Mercator Ice Piedmont", "Miethe Glacier", "Mitterling Glacier",
         "Moby Dick Icefall", "Moczydlowski Glacier", "Moider Glacier", "Mondor Glacier", "Montgolfier Glacier",
         "Montiel Glacier", "Morrison Glacier", "Moser Glacier", "Mouillard Glacier", "Muller Ice Shelf",
         "Nature Conservation Glac", "Nemo Glacier", "Neny Glacier", "Niepce Glacier", "Nobile Glacier",
         "Northeast Glacier", "Nye Glacier", "Orel Ice Fringe", "Otlet Glacier", "Pan Glacier", "Pare Glacier",
         "Penderecki Glacier", "Pequod Glacier", "Perunika Glacier", "Perutz Glacier", "Petzval Glacier",
         "Philippi Rise", "Pimpirev Ice Wall", "Poetry Glacier", "Polar Committee Icefall", "Polar Friendship Glacier",
         "Polonia Piedmont Glacier", "Prespa Glacier", "Professor Glacier", "Prospect Glacier", "Pyke Glacier",
         "Rachel Glacier", "Reid Glacier", "Renard Glacier", "Renaud Glacier", "Robillard Glacier", "Romulus Glacier",
         "Ropotamo Glacier", "Rotch Dome", "Rothschild Island IC,", "Rozier Glacier", "Rudolph Glacier", "Rush Glacier",
         "Russell East Glacier", "Russell West Glacier", "Rybak Glacier", "Sally (Hurd) Glacier", "Sayce Glacier",
         "Seller Glacier", "Shambles Glacier", "Sharp Glacier", "Sheldon Glacier", "Shoesmith Glacier",
         "Sikorsky Glacier", "Simler Snowfield", "Sirocco Glacier", "Sjogren Glacier", "Sleipnir Glacier",
         "Sloman Glacier", "Smyley Island IC,", "Snow Hill IS", "Snow Hill IS,", "Snowshoe Glacier", "Solch Glacier",
         "Somigliana Glacier", "Spaatz Island IC,", "Sphinx Glacier", "Srebarna Glacier", "Starbuck Glacier",
         "Stefan Ice Piedmont", "Stenhouse Glacier", "Strandzha Glacier", "Stubb Glacier", "Stwosz Icefall",
         "Swithinbank Glacier", "Szymanowski Icefall", "Tait Glacier", "Talbot Glacier", "Tarnovo Ice Piedmont",
         "Temple Glacier", "Thunder Glacier", "Todd Glacier", "Tofani Glacier", "Tower Glacier", "Trooz Glacier",
         "Usher Glacier", "Vallot Glacier", "Victory Glacier", "Vogel Glacier", "Wanda Glacier", "Weir Glacier",
         "Wellman Glacier", "Wheatstone Glacier", "Whisky Glacier", "White Eagle Glacier", "Whitecloud Glacier",
         "Widdowson Glacier", "Wiggins Glacier", "Wilkinson Murphy Glacier", "William Glacier", "Windy Glacier",
         "Woodbury Glacier", "Wyspianski Icefall", "Zalewski Glacier", "Zephyr Glacier", "Zonda Glacier"]

glac_names = ["Crane Glacier"]

# create conn to the database
conn = ctd.establish_connection()

for i, glac_name in enumerate(glac_names):

    print("TRY: ", glac_name, i + 1, "of", len(glac_names))

    try:
        import src.sfm_agi.other.get_images_for_glacier as gifg
        glacier_name = glac_name
        image_ids, bounds = gifg.get_images_for_glacier(glacier_name,
                                                        min_overlap_footprint=0.2,
                                                        min_overlap_glacier=0.3,
                                                        return_bounds=True,
                                                        display_footprints=False,
                                                        conn=conn)

        if len(image_ids) == 0:
            print(f"No images found for {glacier_name}")
            continue

        # project settings
        project_name = glac_name
        overwrite = False
        resume = True

        print("TEEEMP")
        project_name = project_name + "_new"

        print(f"Start Project '{project_name}'")
        print("Bounds:", bounds)
        print("Overwrite:", overwrite)
        print("Resume:", resume)

        #if glacier_name is not None and glacier_name != project_name:
        #    raise ValueError("Project name and Glacier name are not the same")

        # accuracy settings (None means not using it)
        camera_accuracy = (100, 100, 100)  # x, y, z in m
        gcp_accuracy = (30, 30, 10)

        # input settings
        limit_images = 0  # 0 means no limit
        use_positions = False  # if true, camera position will be given to agisoft
        use_rotations = True  # if true, camera rotations will be given to agisoft
        only_vertical = False

        # define the path to the image folders
        path_image_folder = "/data/ATM/data_1/aerial/TMA/downloaded"
        path_backup_folder = "/media/fdahle/d3f2d1f5-52c3-4464-9142-3ad7ab1ec06d/data_1/aerial/TMA/downloaded"
        georef_table = "images_extracted"

        # check if we have at least 3 images
        if len(image_ids) < 3:
            raise ValueError("Need at least 3 images")

        # Convert the list to a string that looks like a tuple
        image_ids_string = ', '.join(f"'{image_id}'" for image_id in image_ids)

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

        if data['position_exact'].isnull().sum() > 0:
            for image_id in data[data['position_exact'].isnull()]['image_id']:
                print(image_id + " has no position")
            raise ValueError("Some images have no position")

        # print the number of images without positions
        print(f"Number of images without positions: {data['position_exact'].isnull().sum()}/{len(data)}")
        print(f"Number of images without footprints: {data['footprint_exact'].isnull().sum()}/{len(data)}")

        # remove images without positions or footprints
        print("Removing images without positions or footprints..")
        data = data.dropna(subset=['position_exact', 'footprint_exact'])

        # get the focal length from the military calibration
        data['focal_length'] = data['image_id'].apply(lambda x: lmfl.load_military_focal_length(x, None, conn))

        # replace all values with focal length of 154.25
        #data['focal_length'] = 154.25
        #print(data['focal_length'])
        #print("WAAAAAAAAAAAARNING")

        # print the number of images without focal length
        print(f"Number of images without focal length: {data['focal_length'].isnull().sum()}/{len(data)}")

        # default values based on avg focal lengths of V, L, R
        default_focal_lengths = {'V': 152.07, 'L': 152.78, 'R': 152.29}

        # Define a function to determine the appropriate focal length
        def get_default_focal_length(image_id):
            for key in default_focal_lengths:
                if key in image_id:
                    return default_focal_lengths[key]
            return None  # If none of 'V', 'L', or 'R' is found

        # check if there are any missing focal lengths
        if data['focal_length'].isnull().sum() > 0:
            print("Using default values for focal length..")
            data['focal_length'] = data.apply(
                lambda row: get_default_focal_length(row['image_id']) if pd.isnull(row['focal_length']) else row[
                    'focal_length'],
                axis=1
            )


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

        # check if the images are existing
        import os
        import shutil

        # Define paths
        path_image_folder = "/data/ATM/data_1/aerial/TMA/downloaded"
        backup_image_folder = "/media/fdahle/d3f2d1f5-52c3-4464-9142-3ad7ab1ec06d/data_1/aerial/TMA/downloaded"

        # List of image IDs (e.g., from your data)
        image_ids = data['image_id'].tolist()

        # Create a list with absolute paths
        images_paths = [os.path.join(path_image_folder, image + ".tif") for image in image_ids]

        # Check if the images exist, and copy them from the backup if necessary
        for image_path in images_paths:
            if not os.path.isfile(image_path):
                print(f"Image {image_path} does not exist. Checking backup...")

                # Determine the corresponding path in the backup folder
                relative_path = os.path.relpath(image_path, path_image_folder)
                backup_image_path = os.path.join(backup_image_folder, relative_path)

                if os.path.isfile(backup_image_path):
                    # Create directories if they don't exist
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)

                    # Copy the image from the backup to the target folder
                    print(f"Copying image from backup: {backup_image_path} -> {image_path}")
                    shutil.copy2(backup_image_path, image_path)
                else:
                    # Raise an error if the image doesn't exist in both locations
                    raise FileNotFoundError(f"Image {image_path} does not exist in the target or backup folder")

        # get image shapes for each image
        import src.load.load_image_shape as lis
        shapes_dict = {}
        ids_to_remove = []
        for image_id in image_ids:
            image_shape = lis.load_image_shape(image_id)
            if image_shape is None:
                print(f"Image {image_id} failed to load")
                ids_to_remove.append(image_id)
            else:
                shapes_dict[image_id] = image_shape

        # remove all images that failed to load
        for image_id in ids_to_remove:
            images_paths.remove(os.path.join(path_image_folder, image_id + ".tif"))
            image_ids.remove(image_id)

        # remove from other dicts as well
        dicts = [focal_length_dict, footprint_dict, position_dict, rotation_dict, accuracy_dict, azimuth]
        for d in dicts:
            if d is not None:
                for image_id in ids_to_remove:
                    d.pop(image_id, None)

        # check if the shape of the images is correct
        def _validate_shapes(shapes_dict):
            from collections import defaultdict

            # Group shapes by flight path and direction
            grouped_shapes = defaultdict(list)

            for image_id, shape in shapes_dict.items():
                # Extract flight path and direction from image_id
                flight_path = image_id[2:6]  # Example: extract flight path as substring
                direction = image_id[6:8]  # Example: extract direction as substring
                key = (flight_path, direction)
                grouped_shapes[key].append((image_id, shape))

            # Validate shapes within each group
            for key, images in grouped_shapes.items():
                reference_shape = images[0][1]  # Use the shape of the first image as reference
                ref_height, ref_width = reference_shape

                # iterate all images
                for image_id, shape in images:

                    # get base shape
                    height, width = shape

                    # if the shape is the same, do nothing
                    if abs(ref_height - height) == 0 and abs(ref_width - width) == 0:
                        # do nothing
                        pass

                    # if the shape is almost the same, resize the image
                    elif abs(ref_height - height) < 5 and abs(ref_width - width) < 5:

                        # get the images
                        import src.load.load_image as li
                        img = li.load_image(image_id)

                        enhanced_pth = "/data/ATM/data_1/sfm/agi_data/enhanced/" + image_id + ".tif"
                        if os.path.isfile(enhanced_pth):
                            img_enhanced = li.load_image(enhanced_pth)
                        else:
                            img_enhanced = None

                        # define new shape
                        new_shape = (ref_height, ref_width)

                        # resize the images
                        import src.base.resize_image as ri
                        r_img = ri.resize_image(img, new_shape)
                        if img_enhanced is not None:
                            r_img_enhanced = ri.resize_image(img_enhanced, new_shape)

                        # define save path
                        pth_img_folder = "/data/ATM/data_1/aerial/TMA/downloaded"
                        pth_img = os.path.join(pth_img_folder, image_id + ".tif")

                        # export the images
                        import src.export.export_tiff as et
                        et.export_tiff(r_img, pth_img, overwrite=True, use_lzw=True)
                        if img_enhanced is not None:
                            et.export_tiff(r_img_enhanced, enhanced_pth, overwrite=True, use_lzw=True)

                        sql_string = f"UPDATE images_fid_points SET image_width={ref_width}, " \
                                     f"image_height={ref_height} WHERE image_id='{image_id}'"
                        ctd.execute_sql(sql_string, conn)

                        print("Resized image", image_id, "from", shape, "to", reference_shape)

                    else:

                        print(abs(ref_height - height), abs(ref_width - width))

                        raise ValueError(f"Shape mismatch in group {key}: {image_id} has shape {shape} "
                                         f"but reference shape is {reference_shape}")

        _validate_shapes(shapes_dict)
        print("All shapes are correct")

        ra.run_agi(project_name, images_paths,
                   focal_lengths=focal_length_dict, camera_footprints=footprint_dict,
                   camera_positions=position_dict, camera_rotations=rotation_dict,
                   camera_accuracies=accuracy_dict, gcp_accuracy=gcp_accuracy,
                   azimuth=azimuth, absolute_bounds=bounds,
                   overwrite=overwrite, resume=resume)
    except Exception as e:
        print(e)
        print(f"Failed to create project for glacier {glac_name}")
        continue
